import os
import json
import gzip
import re
import io
import urllib.request
import time


def _has_kind_code(s):
    # 'EP-4013497-B1'(대시 포함), 'EP07075413NWB1'(NW variant) 등 식별자 포맷이
    # 여러 가지라 단순 정규식 하나로는 오탐/누락이 생김 - clean_ep_number()와 동일한
    # 두 단계 스트리핑을 적용해봐서 실제로 뭔가 제거됐으면(=kind code/NW 접미사가
    # 있었으면) kind-coded로 판정
    s2 = str(s).strip().replace('EP', '').replace('-', '')
    stripped = re.sub(r'(NWA|NWB|NW)\d*$', '', s2)
    stripped = re.sub(r'[A-Z]\d*$', '', stripped)
    return stripped != s2 and stripped != ''


def normalize_patent_number(patent_number):
    """서로 다른 출처의 포맷을 같은 키로 조인할 수 있도록 EP 특허번호를 순수 숫자
    문자열로 정규화. 예:
      'EP-4013497-B1' (SureChEMBL 포맷)  -> '4013497'
      'EP3315022A1'   (T_id/P_id 포맷)   -> '3315022'
    숫자를 찾지 못하면 None 반환
    """
    if not patent_number:
        return None
    m = re.match(r'^[A-Za-z]{1,3}-?(\d+)', str(patent_number).strip())
    return m.group(1) if m else None


def get_best_clean_smiles(patcid_entry):
    """patcid_annotations_map[clean_id] 항목에서 PatCID가 이미 인식해둔 고신뢰
    SMILES(class='Clean') 중 confidence가 가장 높은 것을 선택. 없으면 None"""
    if not patcid_entry:
        return None
    candidates = patcid_entry.get('clean_smiles_candidates') or []
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.get('confidence') if c.get('confidence') is not None else -1))


class PatCIDSureChEMBLConnector:
    """PatCID(IBM Deep Search / Nature 2024)와 SureChEMBL(EMBL-EBI / NAR 2016)을 위한 이중 커넥터
    Section C 특허 레코드를 화학 SMILES, InChIKey, PatCID 2D 이미지 주석과 연결

    조회는 순차 레코드 id/파일명이 아니라 실제 EP 특허번호(예: 'EP3315022A1') 기준
    (EPOStreamParser / config.settings.ID_PATENT_MAP_PATH 참고)
    """

    def __init__(self, local_db_dir=None):
        if local_db_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else os.getcwd()
            local_db_dir = os.path.join(base_dir, 'data', 'patcid_surechembl_db')

        self.local_db_dir = os.path.abspath(local_db_dir)
        os.makedirs(self.local_db_dir, exist_ok=True)

        self.patcid_zenodo_doi = 'https://doi.org/10.5281/zenodo.10572870'
        self.surechembl_ftp_url = 'https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/'
        self.surechembl_api_base = 'https://www.ebi.ac.uk/chembl/api/data'

        # 키는 normalize_patent_number()로 정규화한 특허번호
        self.local_chem_map = {}
        self.patcid_annotations_map = {}
        self._load_local_db_dumps()

    def _load_local_db_dumps(self):
        if not os.path.exists(self.local_db_dir):
            return

        # 1. SureChEMBL Map (.txt.gz) 파싱
        try:
            entries = list(os.scandir(self.local_db_dir))
            for entry in entries:
                if entry.name.lower().endswith('.txt.gz'):
                    print(f'Parsing downloaded SureChEMBL Map dump: {entry.name}...')
                    with open(entry.path, 'rb') as f_bin:
                        with gzip.GzipFile(fileobj=f_bin, mode='rb') as gz:
                            with io.TextIOWrapper(gz, encoding='utf-8') as gf:
                                for line in gf:
                                    parts = line.rstrip('\n').split('\t')
                                    if len(parts) >= 5:
                                        schembl_id, smiles, inchikey, field_id, pat_num = parts[0], parts[1], parts[2], parts[3], parts[4]
                                        clean_num = normalize_patent_number(pat_num)
                                        if not clean_num:
                                            continue

                                        if clean_num not in self.local_chem_map:
                                            self.local_chem_map[clean_num] = {
                                                'schembl_id': schembl_id,
                                                'smiles': [smiles],
                                                'inchikey': inchikey,
                                                'patent_number': pat_num
                                            }
                                        else:
                                            if smiles not in self.local_chem_map[clean_num]['smiles']:
                                                self.local_chem_map[clean_num]['smiles'].append(smiles)
                    print(f'Successfully loaded {len(self.local_chem_map)} unique SureChEMBL patent-chemical mappings!')
        except Exception as e:
            print(f'Error loading SureChEMBL dump: {e}')

        # 2. 압축 해제된 PatCID Nature 2024 JSONL 폴더(patcid/) 파싱
        patcid_dir = os.path.join(self.local_db_dir, 'patcid')
        if not os.path.exists(patcid_dir):
            for root, dirs, files in os.walk(self.local_db_dir):
                if 'patcid' in dirs:
                    patcid_dir = os.path.join(root, 'patcid')
                    break

        if os.path.exists(patcid_dir):
            print(f'Parsing unzipped PatCID Nature 2024 dataset folder: {patcid_dir}...')
            epo_patcid = os.path.join(patcid_dir, 'patcid_patent_to_molecules_epo.jsonl')
            if os.path.exists(epo_patcid):
                try:
                    with open(epo_patcid, 'r', encoding='utf-8') as f:
                        for line in f:
                            item = json.loads(line)
                            identifiers = item.get('patent', {}).get('identifiers', [])
                            figures = item.get('figures', [])

                            # class='Clean'(단일 분자로 인식됨) + smiles 필드가 있으면 PatCID가
                            # 이미 자체 OCSR로 계산해둔 값 - 우리가 다시 크롭+인식할 필요 없음.
                            # Markush(가변 치환기 템플릿)/Trash(오탐)는 smiles 없음.
                            # figures[:5]로 자르기 전에(뒤쪽 인덱스에 있을 수 있어) 전체를 훑어야 함
                            clean_smiles_candidates = [
                                {'smiles': fig['smiles']['value'], 'confidence': fig['smiles'].get('confidence'), 'page': fig.get('page')}
                                for fig in figures
                                if fig.get('class') == 'Clean' and isinstance(fig.get('smiles'), dict) and fig['smiles'].get('value')
                            ]

                            # identifiers에는 kind code 포함 별칭('EP9290223A1')과 없는 별칭
                            # ('EP9290223')이 섞여 있고 둘 다 같은 clean_id로 정규화됨. 이 item의
                            # figures/bbox는 kind code가 붙은 특정 공보 기준이라, kind code 없는
                            # 별칭이 나중에 순회되며 덮어쓰면 잘못된 공보를 가리키게 됨(과거 버그) -
                            # kind-coded 별칭을 우선 선택하고 아래에서도 이를 보존
                            preferred_ident = next(
                                (i for i in identifiers if _has_kind_code(i)),
                                (identifiers[0] if identifiers else None)
                            )

                            for ident in identifiers:
                                clean_id = normalize_patent_number(ident)
                                if not clean_id:
                                    continue
                                existing = self.patcid_annotations_map.get(clean_id)
                                if existing and _has_kind_code(existing.get('patent_id', '')) \
                                        and not _has_kind_code(preferred_ident):
                                    continue
                                self.patcid_annotations_map[clean_id] = {
                                    'patent_id': preferred_ident or ident,
                                    'figures': figures[:5],
                                    'figure_count': len(figures),
                                    'has_markush': any(fig.get('class') == 'Markush' for fig in figures),
                                    'clean_smiles_candidates': clean_smiles_candidates,
                                    # 페이지별 실제 크기(인치) - bbox를 픽셀로 환산할 때 필요
                                    # (vision_ocsr/patcid_bbox_cropper.py, 현재 비활성 경로)
                                    'page_dimensions': item.get('page-dimensions', []),
                                    'source': 'PatCID Nature 2024 (IBM Deep Search 33.2GB Dataset)'
                                }
                    print(f'Successfully loaded {len(self.patcid_annotations_map)} PatCID Nature 2024 EPO chemical patent image BBox annotations!')
                except Exception as e:
                    print(f'Error parsing PatCID EPO JSONL: {e}')

    def fetch_surechembl_compound_by_patent(self, patent_id, allow_remote=False):
        """실제 EP 특허번호(예: 'EP3315022A1')로 SMILES/InChIKey 조회

        반환 status:
          - 'LOCAL_DB_MATCH'  : 로컬 SureChEMBL 맵에서 발견
          - 'REMOTE_MATCH'    : SureChEMBL REST API로 발견(allow_remote=True일 때만)
          - 'NOT_FOUND'       : 화학구조 데이터 없음, smiles_list는 빈 리스트
                                 (호출하는 쪽은 NOT_FOUND를 "구조 데이터 없음"으로 처리할 것 -
                                 유사도 0과는 다름)
        """
        clean_id = normalize_patent_number(patent_id)
        if not clean_id:
            return {'status': 'NOT_FOUND', 'surechembl_doc_id': None, 'smiles_list': [], 'inchikey': None,
                    'patent_number': patent_id, 'source': 'No valid patent number provided'}

        match = self.local_chem_map.get(clean_id)
        if match:
            return {
                'status': 'LOCAL_DB_MATCH',
                'surechembl_doc_id': match.get('schembl_id'),
                'smiles_list': match.get('smiles', []),
                'inchikey': match.get('inchikey'),
                'patent_number': match.get('patent_number', patent_id),
                'source': 'Downloaded SureChEMBL Local Map Database'
            }

        if allow_remote:
            url = f'{self.surechembl_api_base}/document/{clean_id}.json'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    smiles_list = data.get('smiles', [])
                    if smiles_list:
                        return {
                            'status': 'REMOTE_MATCH',
                            'surechembl_doc_id': clean_id,
                            'smiles_list': smiles_list,
                            'inchikey': data.get('inchikey'),
                            'patent_number': patent_id,
                            'source': 'SureChEMBL EMBL-EBI REST API'
                        }
            except Exception:
                pass

        return {
            'status': 'NOT_FOUND',
            'surechembl_doc_id': None,
            'smiles_list': [],
            'inchikey': None,
            'patent_number': patent_id,
            'source': 'No SureChEMBL match (local map or remote API)'
        }

    def enrich_patent_record(self, patent_record, allow_remote=False):
        """stream_section_c_records() 레코드를 실제 특허번호(target_patent_id /
        prior_patent_id) 기준으로 보강 (순차 파일명 id 기준 아님)"""
        target_patent_id = patent_record.get('target_patent_id')
        prior_patent_id = patent_record.get('prior_patent_id')

        target_chem = self.fetch_surechembl_compound_by_patent(target_patent_id, allow_remote=allow_remote) if target_patent_id else None
        prior_chem = self.fetch_surechembl_compound_by_patent(prior_patent_id, allow_remote=allow_remote) if prior_patent_id else None

        target_clean = normalize_patent_number(target_patent_id)
        prior_clean = normalize_patent_number(prior_patent_id)
        target_patcid = self.patcid_annotations_map.get(target_clean) if target_clean else None
        prior_patcid = self.patcid_annotations_map.get(prior_clean) if prior_clean else None

        patent_record['target_surechembl_metadata'] = target_chem
        patent_record['prior_surechembl_metadata'] = prior_chem
        patent_record['target_patcid_annotations'] = {
            'has_patcid_mapping': bool(target_patcid),
            'patcid_figures': target_patcid.get('figures', []) if target_patcid else [],
            'has_markush_figures': target_patcid.get('has_markush', False) if target_patcid else False,
        }
        patent_record['prior_patcid_annotations'] = {
            'has_patcid_mapping': bool(prior_patcid),
            'patcid_figures': prior_patcid.get('figures', []) if prior_patcid else [],
            'has_markush_figures': prior_patcid.get('has_markush', False) if prior_patcid else False,
        }
        return patent_record

    def batch_enrich_dataset(self, input_dataset_path, output_dataset_path=None, allow_remote=False):
        if not os.path.exists(input_dataset_path):
            print(f'Input dataset path not found: {input_dataset_path}')
            return False

        with open(input_dataset_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        print(f'Enriching {len(records)} Section C patent records with SureChEMBL & PatCID Dual DB mappings...')
        start_time = time.time()

        enriched_count = 0
        matched_count = 0
        for r in records:
            self.enrich_patent_record(r, allow_remote=allow_remote)
            enriched_count += 1
            if (r.get('target_surechembl_metadata') or {}).get('status') == 'LOCAL_DB_MATCH' or \
               (r.get('prior_surechembl_metadata') or {}).get('status') == 'LOCAL_DB_MATCH':
                matched_count += 1

        elapsed = time.time() - start_time
        print(f'Enrichment completed in {elapsed:.2f}s! Enriched {enriched_count} records, {matched_count} with a real SureChEMBL structure match.')

        if output_dataset_path is None:
            output_dataset_path = input_dataset_path

        with open(output_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f'Enriched SureChEMBL & PatCID dataset saved to: {output_dataset_path}')
        return True
