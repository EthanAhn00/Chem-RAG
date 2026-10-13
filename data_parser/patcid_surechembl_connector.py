import os
import json
import gzip
import re
import io
import urllib.request
import time


def _has_kind_code(s):
    # Identifier formats include 'EP-4013497-B1' and NW variants such as 'EP07075413NWB1'.
    # A single regex can miss or misclassify these; use the same two-stage stripping
    # as clean_ep_number(). If a kind-code/NW suffix was actually removed,
    # classify the identifier as kind-coded.
    s2 = str(s).strip().replace('EP', '').replace('-', '')
    stripped = re.sub(r'(NWA|NWB|NW)\d*$', '', s2)
    stripped = re.sub(r'[A-Z]\d*$', '', stripped)
    return stripped != s2 and stripped != ''


def normalize_patent_number(patent_number):
    """Normalize EP identifiers to numeric strings for cross-source joins.
    Examples: 'EP-4013497-B1' (SureChEMBL) -> '4013497'; 'EP3315022A1' (T_id/P_id) -> '3315022'. Return None when no number is found.
    """
    if not patent_number:
        return None
    m = re.match(r'^[A-Za-z]{1,3}-?(\d+)', str(patent_number).strip())
    return m.group(1) if m else None


def get_best_clean_smiles(patcid_entry):
    """Choose the highest-confidence PatCID SMILES with class='Clean' from patcid_annotations_map[clean_id]. Return None if unavailable."""
    if not patcid_entry:
        return None
    candidates = patcid_entry.get('clean_smiles_candidates') or []
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.get('confidence') if c.get('confidence') is not None else -1))


class PatCIDSureChEMBLConnector:
    """Dual connector for PatCID (IBM Deep Search / Nature 2024) and SureChEMBL (EMBL-EBI / NAR 2016).
    Link Section C records to SMILES, InChIKey, and PatCID 2D image annotations using actual EP publication numbers (e.g. 'EP3315022A1'), not sequential IDs or filenames. See EPOStreamParser and config.settings.ID_PATENT_MAP_PATH.
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

        # Keys are patent numbers normalized by normalize_patent_number().
        self.local_chem_map = {}
        self.patcid_annotations_map = {}
        self._load_local_db_dumps()

    def _load_local_db_dumps(self):
        if not os.path.exists(self.local_db_dir):
            return

        # 1. Parse the SureChEMBL map (.txt.gz).
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

        # 2. Parse the extracted PatCID Nature 2024 JSONL directory (patcid/).
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

                            # class='Clean' with smiles identifies a single molecule already recognized
                            # by PatCID OCSR; no additional cropping/recognition is needed.
                            # Markush templates and Trash false detections have no SMILES.
                            # Scan all figures before truncating to figures[:5]; useful entries may occur later.
                            clean_smiles_candidates = [
                                {'smiles': fig['smiles']['value'], 'confidence': fig['smiles'].get('confidence'), 'page': fig.get('page')}
                                for fig in figures
                                if fig.get('class') == 'Clean' and isinstance(fig.get('smiles'), dict) and fig['smiles'].get('value')
                            ]

                            # Identifiers mix kind-coded aliases ('EP9290223A1') and bare aliases
                            # ('EP9290223'), both normalized to one clean_id. The item's figures/bbox
                            # refer to a particular kind-coded publication; a later bare alias must not
                            # overwrite it and point to the wrong publication (a previous bug).
                            # Prefer kind-coded aliases and preserve this preference below.
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
                                    # Actual page dimensions in inches, needed to convert bounding boxes to pixels.
                                    # Used by vision_ocsr/patcid_bbox_cropper.py (currently inactive).
                                    'page_dimensions': item.get('page-dimensions', []),
                                    'source': 'PatCID Nature 2024 (IBM Deep Search 33.2GB Dataset)'
                                }
                    print(f'Successfully loaded {len(self.patcid_annotations_map)} PatCID Nature 2024 EPO chemical patent image BBox annotations!')
                except Exception as e:
                    print(f'Error parsing PatCID EPO JSONL: {e}')

    def fetch_surechembl_compound_by_patent(self, patent_id, allow_remote=False):
        """Look up SMILES/InChIKey by actual EP publication number (e.g. 'EP3315022A1').
        Statuses:
        - LOCAL_DB_MATCH: found in the local SureChEMBL map.
        - REMOTE_MATCH: found through the REST API, only when allow_remote=True.
        - NOT_FOUND: no structure data; smiles_list is empty. Callers must interpret this as missing structure data, not zero similarity.
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
        """Enrich stream_section_c_records() by target_patent_id / prior_patent_id, not sequential filename IDs."""
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
