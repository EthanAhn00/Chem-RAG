import os
import json
import time
import re
import csv


class EPOStreamParser:
    """Streaming (zero-OOM) reader for large EPO JSON patent datasets.
    Scans labeled subfolders (Simple, Simple_under1021, task3_contrasiv_learning_data) and filters IPC Section C.
    Simple/<id>.json records contain free text and IPC codes but no EP publication numbers, so they cannot independently join XML drawings, SureChEMBL, or PatCID by patent number. Load the original 260207_Target_data.xlsx mapping once and attach its join keys (id -> T_id/P_id patent numbers and T_file/P_file XML filenames) to emitted records.
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

        # Task 1 labeled subfolders. 'Simple'/'Simple_under1021' contain
        # a 'Label' field and can join patent numbers through id_patent_map.csv.
        # 'task3_contrasiv_learning_data' has labels and hard negatives, but no patent-number mapping.
        #
        # Exclude 'Simple_label2'/'Simple_label2_under_1021': neither a 'Label' field
        # nor a corresponding patent-number index is available to verify provenance.
        # Defaulting missing labels to 0 would misrepresent about 4,800 unlabeled records as verified negatives.
        # Keep them in unverified_subfolders and outside Task 1 evaluation until verified.
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
        # Load id -> {T_id, P_id, T_file, P_file, T_IPC, P_IPC} from the reconstructed CSV map.
        # A missing mapping file is not fatal; emit records without patent-number links.
        # Drawing and SureChEMBL/PatCID steps then treat structure/vision data as missing.
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
        # Check whether an IPC code belongs to Section C (chemistry/metallurgy).
        if not ipc_code_str:
            return False
        tokens = re.split(r'[|\s,;]+', str(ipc_code_str))
        for t in tokens:
            t = t.upper().strip()
            if t.startswith(self.section_c_prefixes):
                return True
        return False

    def stream_section_c_records(self, max_samples=20000):
        # Scan target_subfolders and emit deduplicated Section C records.
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

                    # Deduplicate using the beginning of the claim text.
                    dedup_key = hash(target_claim[:100] + prior_claim[:100])
                    if dedup_key in seen_keys:
                        continue

                    # IPC field names vary: Simple variants use 'Target_IPC'/'Prior_IPC';
                    # task3_contrasiv_learning_data uses only 'T_IPC' (no Prior_IPC/P_IPC).
                    target_ipc = data.get('Target_IPC') or data.get('T_IPC') or ''
                    prior_ipc = data.get('Prior_IPC') or data.get('P_IPC') or ''

                    is_target_c = self.is_section_c_patent(target_ipc)
                    is_prior_c = self.is_section_c_patent(prior_ipc)

                    if is_target_c or is_prior_c:
                        seen_keys.add(dedup_key)

                        # Recover actual EP publication numbers/XML filenames from the record ID.
                        # Example: '10001.json' -> id 10001 -> {T_id: EP3649854A1, P_id: EP3315022A1, ...}.
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
                            'label_verified': 'Label' in data,  # False indicates a defaulted label; this should not occur in target_subfolders.
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
