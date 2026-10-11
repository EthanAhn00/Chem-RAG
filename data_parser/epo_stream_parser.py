import os
import json
import time
import re
import csv


class EPOStreamParser:
    """대용량 EPO JSON 특허 데이터셋을 위한 스트림 방식(zero-OOM) 리더
    골드 라벨이 있는 서브폴더(Simple, Simple_under1021, task3_contrasiv_learning_data)를
    스캔하고 IPC Section C로 필터링

    각 Simple/<id>.json 레코드는 자유 텍스트 + IPC 코드만 담고 실제 EP 공개번호는
    없어, 단독으로는 XML 도면 파일/SureChEMBL/PatCID와 특허번호 기준 조인 불가능
    원본 `260207_Target_data.xlsx`에 그 조인 키(id -> T_id/P_id 특허번호,
    T_file/P_file XML 파일명)가 있어, 한 번 로드해 모든 레코드에 붙여서 내보냄
    """

    def __init__(self, epo_parsed_base=None, id_patent_map_path=None):
        if epo_parsed_base is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            external_root = os.environ.get(
                'CHEMRAG_EXTERNAL_DATA_ROOT', os.path.join(project_dir, 'data', 'external')
            )
            epo_parsed_base = os.environ.get(
                'CHEMRAG_EPO_PARSED_BASE', os.path.join(external_root, 'EPO_SR_parsed_260207')
            )
        self.epo_parsed_base = epo_parsed_base

        # Task 1 벤치마크용 골드 라벨 서브폴더. 'Simple'/'Simple_under1021'은 실제
        # 'Label' 필드 보유 + id_patent_map.csv로 특허번호 조인 가능. 'task3_contrasiv_
        # learning_data'도 실제 'Label' 필드 보유(하드 네거티브 포함)하나 특허번호 연결은 없음.
        #
        # 'Simple_label2'/'Simple_label2_under_1021'은 제외: 'Label' 필드 자체가 없고
        # 대응하는 특허번호 인덱스도 없어 프로버넌스 확인 불가. 결측 라벨을 0으로
        # 기본값 처리하면 약 4,800건의 라벨 없는 데이터가 "검증된 negative"로 둔갑되므로,
        # 확인 전까지는 `unverified_subfolders`로 분리해 Task 1 평가셋에서 제외
        self.target_subfolders = [
            'Simple',
            'Simple_under1021',
            'task3_contrasiv_learning_data',
        ]
        self.unverified_subfolders = [
            'Simple_label2',
            'Simple_label2_under_1021',
        ]
        self.section_c_prefixes = ('C', 'A61K', 'A61P')

        if id_patent_map_path is None:
            try:
                from config.settings import ID_PATENT_MAP_PATH
                id_patent_map_path = ID_PATENT_MAP_PATH
            except Exception:
                id_patent_map_path = None

        self.id_patent_map = self._load_id_patent_map(id_patent_map_path)

    def _load_id_patent_map(self, path):
        # 재구축한 CSV 매핑에서 id -> {T_id, P_id, T_file, P_file, T_IPC, P_IPC} 로드
        # 매핑 파일이 없어도 치명적이지 않음 - 레코드는 특허번호 연결 없이 반환되고
        # 이후 단계(도면 조회, SureChEMBL/PatCID 보강)에서 "구조/비전 데이터 없음"으로 처리
        id_map = {}
        if not path or not os.path.exists(path):
            print(f'[EPOStreamParser] id_patent_map not found at {path} - records will have no patent-number linkage.')
            return id_map

        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rid = int(row['id'])
                except (KeyError, ValueError, TypeError):
                    continue
                id_map[rid] = {
                    'target_patent_id': row.get('T_id') or None,
                    'prior_patent_id': row.get('P_id') or None,
                    'target_xml_file': row.get('T_file') or None,
                    'prior_xml_file': row.get('P_file') or None,
                }
        print(f'[EPOStreamParser] Loaded {len(id_map)} id -> patent-number/XML-filename mappings.')
        return id_map

    def is_section_c_patent(self, ipc_code_str):
        # IPC 분류 코드의 Section C(화학·야금) 소속 여부 확인
        if not ipc_code_str:
            return False
        tokens = re.split(r'[|\s,;]+', str(ipc_code_str))
        for t in tokens:
            t = t.upper().strip()
            if t.startswith(self.section_c_prefixes):
                return True
        return False

    def stream_section_c_records(self, max_samples=20000):
        # target_subfolders를 스캔해 중복 제거된 고유 Section C 레코드 추출
        if not os.path.exists(self.epo_parsed_base):
            print(f'EPO Parsed Base Directory not found: {self.epo_parsed_base}')
            return []

        section_c_records = []
        seen_keys = set()
        start_time = time.time()

        total_scanned_files = 0
        linked_count = 0

        for sub in self.target_subfolders:
            sub_path = os.path.join(self.epo_parsed_base, sub)
            if not os.path.exists(sub_path):
                continue

            json_files = [f for f in os.listdir(sub_path) if f.endswith('.json')]
            total_scanned_files += len(json_files)
            print(f'Scanning subfolder [{sub}]: {len(json_files)} JSON records...')

            for f in json_files:
                file_path = os.path.join(sub_path, f)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as jf:
                        data = json.load(jf)

                    target_claim = data.get('Target', '').strip()
                    prior_claim = data.get('Prior', '').strip()

                    # 청구항 텍스트 앞부분 기준 중복 제거 키
                    dedup_key = hash(target_claim[:100] + prior_claim[:100])
                    if dedup_key in seen_keys:
                        continue

                    # IPC 필드명은 서브폴더마다 다름: 'Simple'류는 'Target_IPC'/'Prior_IPC',
                    # 'task3_contrasiv_learning_data'는 'T_IPC'만 사용(Prior_IPC/P_IPC 없음)
                    target_ipc = data.get('Target_IPC') or data.get('T_IPC') or ''
                    prior_ipc = data.get('Prior_IPC') or data.get('P_IPC') or ''

                    is_target_c = self.is_section_c_patent(target_ipc)
                    is_prior_c = self.is_section_c_patent(prior_ipc)

                    if is_target_c or is_prior_c:
                        seen_keys.add(dedup_key)

                        # 레코드 id로 실제 EP 특허번호/XML 파일명 복원
                        # 예: "10001.json" -> id 10001 -> {T_id: EP3649854A1, P_id: EP3315022A1, ...}
                        record_id = None
                        stem = os.path.splitext(f)[0]
                        if stem.isdigit():
                            record_id = int(stem)
                        link = self.id_patent_map.get(record_id, {}) if record_id is not None else {}
                        if link.get('target_patent_id') or link.get('prior_patent_id'):
                            linked_count += 1

                        record = {
                            'filename': f,
                            'subfolder': sub,
                            'record_id': record_id,
                            'target_claim': target_claim,
                            'prior_claim': prior_claim,
                            'ground_truth_label': int(data.get('Label', 0)),
                            'label_verified': 'Label' in data,  # False면 라벨 기본값 처리됨 - target_subfolders에서는 발생 안 해야 함
                            'target_ipc': target_ipc,
                            'prior_ipc': prior_ipc,
                            'is_target_c': is_target_c,
                            'is_prior_c': is_prior_c,
                            'target_patent_id': link.get('target_patent_id'),
                            'prior_patent_id': link.get('prior_patent_id'),
                            'target_xml_file': link.get('target_xml_file'),
                            'prior_xml_file': link.get('prior_xml_file'),
                        }
                        section_c_records.append(record)

                    if len(section_c_records) >= max_samples:
                        break
                except Exception:
                    pass

            if len(section_c_records) >= max_samples:
                break

        elapsed = time.time() - start_time
        print(f'Multi-folder stream scanning completed in {elapsed:.2f}s!')
        print(f'Scanned {total_scanned_files} total JSON files across {len(self.target_subfolders)} subfolders ({", ".join(self.target_subfolders)}) -> Extracted {len(section_c_records)} UNIQUE Section C Chemical Patent Pairs.')
        print(f'Records with recovered patent-number linkage: {linked_count}/{len(section_c_records)}')
        return section_c_records
