"""특허-화학구조 위치 메타데이터(in_claim / in_figure / 등장 횟수) 생성 스크립트.

원본 CSV는 절대 수정하지 않고, data/db_export/derived/ 아래에 재사용 가능한
산출물을 새로 만든다.

핵심 원칙 - 데이터에 근거하지 않은 값을 만들지 않는다
  * in_claim / in_figure는 true / false / unknown 3-값. '확인 못 함'은 false가 아니라 unknown.
  * in_claim=true는 아래 중 하나의 추적 가능한 근거가 있을 때만 부여한다.
      1) patent_molecule의 명시적 claim 위치 연결(location/unit_ref)
      2) claim_element.element_smiles의 RDKit 검증된 구조 일치
      3) 청구항 텍스트에서 추출한 InChIKey / SMILES의 검증된 일치
    단순 문자열 부분일치는 쓰지 않는다 - 반드시 RDKit으로 파싱해 canonical SMILES
    또는 InChIKey 수준에서 비교한다.
  * in_figure=true는 figure.ocsr_smiles 등 검증된 figure-molecule 연결이 있을 때만.
    '도면 행이 존재한다'는 사실만으로는 절대 true로 만들지 않는다.
  * false는 해당 특허에 구조 주석이 실제로 존재하는데(=닫힌 세계 가정이 가능한데)
    이 분자가 거기 없을 때만 부여한다.
  * claim_occurrence_count는 검증 가능한 범위에서만 정수, 아니면 null + 상태 필드.

사용법:
    python3 scripts/qdrant/qdrant_build_structure_metadata.py
    python3 scripts/qdrant/qdrant_build_structure_metadata.py --out-dir data/db_export/derived
"""
import os
import re
import sys
import csv
import glob
import json
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
csv.field_size_limit(10 ** 9)

from config.settings import DATA_DIR
from qdrant_engine.schemas import TRUE, FALSE, UNKNOWN
from qdrant_engine.provenance import source_version

DB_EXPORT_DIR = os.path.join(DATA_DIR, 'db_export')
DEFAULT_OUT_DIR = os.path.join(DB_EXPORT_DIR, 'derived')

# 청구항 텍스트에서 InChIKey는 형식이 고정돼 있어 오탐 위험이 매우 낮다
INCHIKEY_RE = re.compile(r'\b([A-Z]{14}-[A-Z]{10}-[A-Z])\b')
# SMILES 후보 토큰: 화학식에 쓰이는 문자만으로 이루어지고 결합/고리 기호를 포함한 것
SMILES_CANDIDATE_RE = re.compile(r'(?<![A-Za-z0-9])([A-Za-z0-9@\+\-\[\]\(\)=#\\/%\.]{6,})(?![A-Za-z0-9])')
# 최소 중원자 수 - 너무 작은 분자는 우연한 문자열 일치 위험이 커서 텍스트 근거로 인정하지 않음
MIN_HEAVY_ATOMS_FOR_TEXT_EVIDENCE = 3


def load_csv(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def nonempty(v):
    return isinstance(v, str) and v.strip() != ''


class Canonicalizer:
    """SMILES -> (canonical SMILES, InChIKey, 중원자 수). 실패는 None."""

    def __init__(self):
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        self._Chem = Chem
        self._cache = {}

    def canon(self, smiles):
        if not nonempty(smiles):
            return None
        if smiles in self._cache:
            return self._cache[smiles]
        result = None
        try:
            mol = self._Chem.MolFromSmiles(smiles)
            if mol is not None:
                result = {
                    'canonical_smiles': self._Chem.MolToSmiles(mol),
                    'inchikey': self._Chem.MolToInchiKey(mol) or None,
                    'heavy_atoms': mol.GetNumHeavyAtoms(),
                }
        except Exception:
            result = None
        self._cache[smiles] = result
        return result


def extract_text_structure_evidence(text, canon, target_canonical, target_inchikey):
    """청구항 텍스트에서 '검증된' 구조 근거만 뽑는다.

    반환: (근거 리스트, 텍스트 스캔 여부)
    - InChIKey 완전 일치
    - SMILES 후보 토큰을 RDKit으로 파싱해 canonical SMILES가 대상과 같은 경우
    """
    evidence = []
    if not nonempty(text):
        return evidence, False

    if target_inchikey:
        for m in INCHIKEY_RE.finditer(text):
            if m.group(1).upper() == target_inchikey.upper():
                evidence.append({'method': 'claims_text_inchikey', 'matched': m.group(1),
                                 'char_offset': m.start()})

    for m in SMILES_CANDIDATE_RE.finditer(text):
        token = m.group(1)
        # 순수 영단어/숫자는 SMILES 후보에서 제외(결합·고리·분기 기호가 하나도 없으면 스킵)
        if not any(ch in token for ch in '()[]=#@+.\\/123456789'):
            continue
        parsed = canon.canon(token)
        if not parsed:
            continue
        if parsed['heavy_atoms'] < MIN_HEAVY_ATOMS_FOR_TEXT_EVIDENCE:
            continue
        if parsed['canonical_smiles'] == target_canonical:
            evidence.append({'method': 'claims_text_smiles', 'matched': token,
                             'char_offset': m.start()})
    return evidence, True


def load_ocsr_cache(cache_dir, canon):
    """data/ocsr_cache/*.json 의 OCSR 성공 결과를 특허번호 기준으로 정리한다.

    이 결과는 도면 이미지에서 인식된 구조지만 fig_id가 아니라 문서 페이지 번호만
    가지고 있다. figure.csv의 fig_no/fig_id와의 대응이 검증되지 않았으므로
    '검증된 figure 근거'로 승격하지 않고 별도의 미검증 근거로 취급한다.
    """
    entries = defaultdict(list)
    stats = {'files': 0, 'success': 0, 'not_found': 0, 'unparseable_smiles': 0}
    if not os.path.isdir(cache_dir):
        return entries, stats
    for path in sorted(glob.glob(os.path.join(cache_dir, '*.json'))):
        stats['files'] += 1
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rec = json.load(f)
        except Exception:
            continue
        if rec.get('status') != 'SUCCESS':
            stats['not_found'] += 1
            continue
        stats['success'] += 1
        parsed = canon.canon(rec.get('smiles'))
        if not parsed:
            stats['unparseable_smiles'] += 1
            continue
        key = normalize_patent_key(os.path.basename(path)[:-5])
        entries[key].append({
            'canonical': parsed['canonical_smiles'], 'inchikey': parsed['inchikey'],
            'source_page': rec.get('source_page'), 'confidence': rec.get('confidence'),
            'engine': rec.get('engine'), 'crop_method': rec.get('crop_method'),
        })
    return entries, stats


def _patcid_candidate_lines(jsonl_path, prefixes):
    """4GB급 JSONL에서 관심 특허번호가 포함된 줄만 뽑는다.

    파이썬 루프로 전수 스캔하면 매우 느려서, 가능하면 grep -F로 1차 필터링하고
    grep이 없으면 컴파일된 정규식 하나로 대체한다(정확도는 동일, 속도만 차이).
    """
    import shutil
    import tempfile
    import subprocess

    if shutil.which('grep'):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write('\n'.join(sorted(prefixes)) + '\n')
            pat_file = f.name
        try:
            proc = subprocess.run(['grep', '-F', '-f', pat_file, jsonl_path],
                                  capture_output=True, text=True, timeout=600)
            # grep 종료코드 1 = 매칭 없음(정상)
            if proc.returncode in (0, 1):
                return [l for l in proc.stdout.splitlines() if l.strip()], 'grep'
        except Exception:
            pass
        finally:
            try:
                os.remove(pat_file)
            except OSError:
                pass

    rx = re.compile('|'.join(re.escape(k) for k in sorted(prefixes)))
    out = []
    with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if rx.search(line):
                out.append(line)
    return out, 'python-regex'


def load_patcid_figures(jsonl_path, wanted_keys, canon, max_lines=None):
    """PatCID EPO 매핑에서 해당 특허들의 Clean 구조를 읽는다(그대로는 미검증 근거).

    PatCID는 페이지/bbox 단위 인식 결과라 figure.csv의 fig_id와 1:1 대응이
    검증되지 않았다. 따라서 in_figure=true의 근거로 쓰지 않고 별도 기록만 한다.
    """
    entries = defaultdict(list)
    stats = {'file_present': os.path.exists(jsonl_path), 'prefilter': None,
             'candidate_lines': 0, 'patents_matched': 0,
             'clean_structures': 0, 'markush_structures': 0, 'false_positive_lines': 0}
    if not stats['file_present']:
        return entries, stats

    prefixes = set(wanted_keys)
    lines, how = _patcid_candidate_lines(jsonl_path, prefixes)
    stats['prefilter'] = how
    stats['candidate_lines'] = len(lines)

    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        ids = obj.get('patent', {}).get('identifiers', []) or []
        keys = {normalize_patent_key(x) for x in ids} & prefixes
        if not keys:
            stats['false_positive_lines'] += 1   # 부분 문자열 우연 일치
            continue
        stats['patents_matched'] += 1
        for fig in obj.get('figures', []) or []:
            cls = fig.get('class')
            if cls == 'Markush':
                stats['markush_structures'] += 1
            if cls != 'Clean' or not fig.get('smiles'):
                continue
            parsed = canon.canon((fig.get('smiles') or {}).get('value'))
            if not parsed:
                continue
            stats['clean_structures'] += 1
            for k in keys:
                entries[k].append({
                    'canonical': parsed['canonical_smiles'], 'inchikey': parsed['inchikey'],
                    'page': fig.get('page'), 'bbox': fig.get('bbox'),
                    'confidence': (fig.get('smiles') or {}).get('confidence'),
                    'figure_class': cls,
                })
    return entries, stats


def normalize_patent_key(patent_id):
    """'EP3660125A1', 'EP3660125', 'EP-3660125-B1' -> 'EP3660125' (kind code 무시)."""
    m = re.match(r'^([A-Za-z]{2})-?(\d+)', str(patent_id or '').strip())
    return f'{m.group(1).upper()}{m.group(2)}' if m else str(patent_id or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--ocsr-cache-dir', default=os.path.join(DATA_DIR, 'ocsr_cache'))
    ap.add_argument('--patcid-epo-jsonl', default=None,
                    help='PatCID EPO 매핑 jsonl 경로(기본: data/patcid_surechembl_db/patcid/...)')
    ap.add_argument('--skip-patcid', action='store_true', help='PatCID 스캔 건너뛰기(대용량)')
    ap.add_argument('--limit-patents', type=int, default=None, help='디버그용 - 처리할 특허 수 상한')
    args = ap.parse_args()

    paths = {name: os.path.join(DB_EXPORT_DIR, f'{name}.csv') for name in
             ('molecule', 'patent_molecule', 'patent_content', 'claim_element', 'figure')}
    src_version, src_detail = source_version(list(paths.values()))
    print(f'[source_version] {src_version} {src_detail}')

    print('[1/4] CSV 로딩...')
    molecules = {r['mol_id']: r for r in load_csv(paths['molecule'])}
    pm_rows = load_csv(paths['patent_molecule'])
    claims_text_of = {}
    for r in load_csv(paths['patent_content']):
        claims_text_of[r['patent_id']] = r.get('claims_text') or ''

    # data/epo_txt의 특허 전문 청구항으로 claims_text를 보강한다.
    # patent_content.claims_text가 비어 있는 특허만 채우며, 기존 값은 덮어쓰지 않는다.
    epo_txt_dir = os.path.join(DATA_DIR, 'epo_txt')
    epo_txt_filled = 0
    if os.path.isdir(epo_txt_dir):
        for fn in sorted(os.listdir(epo_txt_dir)):
            if not (fn.startswith('epo_txt_') and fn.endswith('.json')):
                continue
            pid = fn[len('epo_txt_'):-len('.json')]
            if nonempty(claims_text_of.get(pid)):
                continue
            try:
                with open(os.path.join(epo_txt_dir, fn), encoding='utf-8') as f:
                    doc = json.load(f)
            except Exception:
                continue
            txt = '\n'.join((cl.get('text') or '') for cl in (doc.get('claims') or []))
            if txt.strip():
                claims_text_of[pid] = txt
                epo_txt_filled += 1
    print(f'  epo_txt로 claims_text 보강: {epo_txt_filled}건')

    # 알려진 문제: claim_element.element_id에 중복이 있다. 내부 식별자는 행 번호를 붙여
    # 충돌 없이 만들고, 원본 element_id는 근거(payload)에 그대로 남긴다.
    claim_elements_by_patent = defaultdict(list)
    for i, r in enumerate(load_csv(paths['claim_element'])):
        claim_elements_by_patent[r['patent_id']].append({
            'internal_id': f"{r['element_id']}#row{i}",
            'element_id': r['element_id'],
            'claim_no': r.get('claim_no') or None,
            'element_no': r.get('element_no') or None,
            'element_text': r.get('element_text') or '',
            'element_smiles': r.get('element_smiles') or '',
        })

    figures_by_patent = defaultdict(list)
    for r in load_csv(paths['figure']):
        figures_by_patent[r['patent_id']].append({
            'fig_id': r['fig_id'], 'fig_no': r.get('fig_no'),
            'ocsr_smiles': r.get('ocsr_smiles') or '',
            'image_path': r.get('image_path') or '',
        })

    # (patent_id, mol_id)가 원본에 몇 행으로 기록됐는지(= DB 내 중복 등장 횟수).
    # 청구항 안에서 몇 번 나오는지(claim_occurrence_count)와는 다른 값이다.
    pair_row_counts = Counter((r['patent_id'], r['mol_id']) for r in pm_rows)

    # (patent_id, mol_id) 중복 제거 - 원본에는 중복 행이 있다
    seen_pairs = set()
    pairs = []
    duplicate_rows = 0
    for r in pm_rows:
        key = (r['patent_id'], r['mol_id'])
        if key in seen_pairs:
            duplicate_rows += 1
            continue
        seen_pairs.add(key)
        pairs.append(r)
    print(f'  patent_molecule 원본 {len(pm_rows)}행 -> 고유 (patent_id, mol_id) {len(pairs)}쌍 '
          f'(중복 {duplicate_rows}행 제거)')

    canon = Canonicalizer()

    # 특허별로 '구조 주석이 존재하는지'를 먼저 판정한다(닫힌 세계 가정 가능 여부)
    patent_has_claim_smiles_annotation = {
        pid: any(nonempty(e['element_smiles']) for e in els)
        for pid, els in claim_elements_by_patent.items()}
    patent_has_figure_ocsr_annotation = {
        pid: any(nonempty(f['ocsr_smiles']) for f in figs)
        for pid, figs in figures_by_patent.items()}

    print('[2/4] 청구항 구조 주석 canonical 인덱스 구성...')
    claim_struct_index = defaultdict(list)   # patent_id -> [(canonical, inchikey, element_meta)]
    for pid, els in claim_elements_by_patent.items():
        for e in els:
            if not nonempty(e['element_smiles']):
                continue
            parsed = canon.canon(e['element_smiles'])
            if parsed:
                claim_struct_index[pid].append((parsed['canonical_smiles'], parsed['inchikey'], e))

    figure_struct_index = defaultdict(list)  # patent_id -> [(canonical, inchikey, figure_meta)]
    for pid, figs in figures_by_patent.items():
        for fmeta in figs:
            if not nonempty(fmeta['ocsr_smiles']):
                continue
            parsed = canon.canon(fmeta['ocsr_smiles'])
            if parsed:
                figure_struct_index[pid].append((parsed['canonical_smiles'], parsed['inchikey'], fmeta))

    wanted_keys = {normalize_patent_key(pid) for pid in {r['patent_id'] for r in pairs}}
    print('[3/5] 추가 근거 소스 조사 (OCSR 캐시 / PatCID)...')
    ocsr_entries, ocsr_stats = load_ocsr_cache(args.ocsr_cache_dir, canon)
    print(f'  OCSR 캐시: {ocsr_stats}')
    print(f'  OCSR 성공 결과가 있는 특허 중 patent_molecule에 존재: '
          f'{len(set(ocsr_entries) & wanted_keys)}건')
    patcid_path = args.patcid_epo_jsonl or os.path.join(
        DATA_DIR, 'patcid_surechembl_db', 'patcid', 'patcid_patent_to_molecules_epo.jsonl')
    if args.skip_patcid:
        patcid_entries, patcid_stats = defaultdict(list), {'skipped': True}
    else:
        patcid_entries, patcid_stats = load_patcid_figures(patcid_path, wanted_keys, canon)
    print(f'  PatCID: {patcid_stats}')

    print('[4/5] (patent, molecule)별 위치 판정...')
    records = []
    patents_seen = set()
    for r in pairs:
        pid, mid = r['patent_id'], r['mol_id']
        if args.limit_patents and pid not in patents_seen and len(patents_seen) >= args.limit_patents:
            continue
        patents_seen.add(pid)

        mol = molecules.get(mid, {})
        smiles = mol.get('smiles') or ''
        parsed = canon.canon(smiles)
        target_canonical = parsed['canonical_smiles'] if parsed else None
        target_inchikey = (parsed['inchikey'] if parsed else None) or (mol.get('inchikey') or None)

        evidence = {'claim': [], 'figure': [], 'unverified': [],
                    'methods_attempted': [], 'notes': []}
        pkey = normalize_patent_key(pid)

        # ---------------- 1) patent_molecule의 명시적 위치 연결 ----------------
        evidence['methods_attempted'].append('patent_molecule_location')
        loc = (r.get('location') or '').strip().lower()
        unit_ref = (r.get('unit_ref') or '').strip()
        if loc in ('claim', 'claims'):
            evidence['claim'].append({'method': 'patent_molecule_location', 'unit_ref': unit_ref or None})
        elif loc in ('figure', 'fig', 'drawing'):
            evidence['figure'].append({'method': 'patent_molecule_location', 'unit_ref': unit_ref or None})

        # ---------------- 2) claim_element.element_smiles 검증 일치 ----------------
        evidence['methods_attempted'].append('claim_element_smiles')
        matched_claim_nos = []
        if target_canonical:
            for cand_canon, cand_ik, meta in claim_struct_index.get(pid, []):
                same = (cand_canon == target_canonical) or (
                    target_inchikey and cand_ik and cand_ik == target_inchikey)
                if same:
                    evidence['claim'].append({
                        'method': 'claim_element_smiles',
                        'element_id': meta['element_id'],
                        'internal_id': meta['internal_id'],
                        'claim_no': meta['claim_no'],
                    })
                    if meta['claim_no']:
                        matched_claim_nos.append(meta['claim_no'])

        # ---------------- 3) 청구항 텍스트에서 검증된 구조 추출 ----------------
        evidence['methods_attempted'].append('claims_text_extraction')
        claims_text = claims_text_of.get(pid, '')
        text_evidence, text_scanned = ([], False)
        if target_canonical:
            text_evidence, text_scanned = extract_text_structure_evidence(
                claims_text, canon, target_canonical, target_inchikey)
        evidence['claim'].extend(text_evidence)

        # ---------------- 4) figure.ocsr_smiles 검증 일치 ----------------
        evidence['methods_attempted'].append('figure_ocsr_smiles')
        matched_fig_ids = []
        if target_canonical:
            for cand_canon, cand_ik, fmeta in figure_struct_index.get(pid, []):
                same = (cand_canon == target_canonical) or (
                    target_inchikey and cand_ik and cand_ik == target_inchikey)
                if same:
                    evidence['figure'].append({'method': 'figure_ocsr_smiles', 'fig_id': fmeta['fig_id'],
                                               'fig_no': fmeta['fig_no']})
                    matched_fig_ids.append(fmeta['fig_id'])

        # ------- 5) OCSR 캐시(도면 이미지 인식 결과) - 페이지 단위, fig_id 미검증 -------
        evidence['methods_attempted'].append('ocsr_cache')
        if target_canonical:
            for e in ocsr_entries.get(pkey, []):
                if e['canonical'] == target_canonical or (
                        target_inchikey and e['inchikey'] and e['inchikey'] == target_inchikey):
                    evidence['unverified'].append({
                        'method': 'ocsr_cache', 'source_page': e['source_page'],
                        'confidence': e['confidence'], 'engine': e['engine'],
                        'crop_method': e['crop_method'],
                        'why_unverified': 'source_page가 figure.csv의 fig_id/fig_no와 대응되는지 검증되지 않음',
                    })

        # ------- 6) PatCID Clean 구조 - 페이지/bbox 단위, fig_id 미검증 -------
        evidence['methods_attempted'].append('patcid_clean_figures')
        if target_canonical:
            for e in patcid_entries.get(pkey, []):
                if e['canonical'] == target_canonical or (
                        target_inchikey and e['inchikey'] and e['inchikey'] == target_inchikey):
                    evidence['unverified'].append({
                        'method': 'patcid_clean_figure', 'page': e['page'], 'bbox': e['bbox'],
                        'confidence': e['confidence'], 'figure_class': e['figure_class'],
                        'why_unverified': 'PatCID page/bbox의 의미가 figure.csv fig_id와 대응되는지 검증되지 않음',
                    })

        # ---------------- 판정 ----------------
        has_claim_annotation = patent_has_claim_smiles_annotation.get(pid, False)
        has_figure_annotation = patent_has_figure_ocsr_annotation.get(pid, False)

        if evidence['claim']:
            in_claim = TRUE
        elif has_claim_annotation:
            # 이 특허는 청구항 구조 주석이 실제로 있는데 이 분자가 없다 -> 근거 있는 false
            in_claim = FALSE
        else:
            in_claim = UNKNOWN

        if evidence['figure']:
            in_figure = TRUE
        elif has_figure_annotation:
            in_figure = FALSE
        else:
            in_figure = UNKNOWN

        # ---- 등장 횟수 ----
        # claim_occurrence_count      : 실제로 매칭된 occurrence 수(같은 청구항 안 반복도 각각 1회)
        # claim_count_with_structure  : 그 구조가 한 번 이상 등장한 '고유 청구항' 수
        # 존재만 알고 횟수를 셀 수 없는 근거(patent_molecule.location=claim 등)는
        # 1로 추정하지 않고 null + known_presence_unknown_count 로 남긴다.
        countable = [e for e in evidence['claim']
                     if e.get('method') in ('claim_element_smiles', 'claims_text_smiles',
                                            'claims_text_inchikey')]
        presence_only = [e for e in evidence['claim']
                         if e.get('method') == 'patent_molecule_location']
        distinct_claims = sorted({str(e['claim_no']) for e in evidence['claim']
                                  if e.get('claim_no')})

        if in_claim == TRUE:
            if countable:
                count = len(countable)
                claim_count_with_structure = len(distinct_claims) if distinct_claims else None
                occurrence_status = ('verified_occurrences' if distinct_claims
                                     else 'verified_occurrences_without_claim_no')
            else:
                # 존재 근거만 있고 횟수를 셀 수 없는 경우
                count = None
                claim_count_with_structure = len(distinct_claims) or None
                occurrence_status = 'known_presence_unknown_count'
        elif in_claim == FALSE:
            count, claim_count_with_structure, occurrence_status = 0, 0, 'verified_absent'
        else:
            count, claim_count_with_structure = None, None
            occurrence_status = ('unverifiable_no_claims_text' if not nonempty(claims_text)
                                 else 'unverifiable_no_claim_structure_annotation')

        occurrence_evidence = [
            {'method': e.get('method'), 'element_id': e.get('element_id'),
             'claim_no': e.get('claim_no'), 'char_offset': e.get('char_offset'),
             'matched': e.get('matched'), 'match_basis': 'canonical_smiles_or_inchikey',
             'status': 'verified'}
            for e in countable
        ] + [
            {'method': e.get('method'), 'unit_ref': e.get('unit_ref'),
             'match_basis': 'explicit_location_annotation', 'status': 'presence_only'}
            for e in presence_only
        ]

        # 상태 요약 - 왜 unknown인지 이유가 남도록
        if in_claim == TRUE or in_figure == TRUE:
            status = 'verified_' + '_'.join(
                [x for x in (('claim' if in_claim == TRUE else None),
                             ('figure' if in_figure == TRUE else None)) if x])
        elif in_claim == FALSE or in_figure == FALSE:
            status = 'annotated_absent'
        elif not target_canonical:
            status = 'unknown_molecule_smiles_unparseable'
        elif evidence['unverified']:
            methods = sorted({e['method'] for e in evidence['unverified']})
            status = 'unknown_unverified_figure_evidence_' + '_'.join(methods)
        elif text_scanned:
            status = 'unknown_text_scanned_no_verified_match'
        elif not nonempty(claims_text):
            status = 'unknown_no_claims_text'
        else:
            status = 'unknown_no_location_annotation'

        if not has_claim_annotation:
            evidence['notes'].append('claim_element.element_smiles가 이 특허에 비어 있어 청구항 구조 대조 불가')
        if not has_figure_annotation:
            evidence['notes'].append('figure.ocsr_smiles가 이 특허에 비어 있어 도면 구조 대조 불가')

        records.append({
            'patent_id': pid,
            'mol_id': mid,
            'smiles': smiles,
            'canonical_smiles': target_canonical,
            'inchikey': target_inchikey,
            'source': mol.get('source') or None,
            'in_claim': in_claim,
            'in_figure': in_figure,
            'claim_occurrence_count': count,
            'claim_count_with_structure': claim_count_with_structure,
            'claim_occurrence_status': occurrence_status,
            'claim_unit_refs': distinct_claims,
            'occurrence_evidence': occurrence_evidence,
            'figure_unit_refs': matched_fig_ids,
            # DB 내 중복 등장 횟수: patent_molecule.csv에 같은 (patent_id, mol_id)가 몇 행인지.
            # 청구항 본문 등장 횟수가 아니라 레코드 중복 수다.
            'db_occurrence_count': pair_row_counts[(pid, mid)],
            # 이 특허에 도면(figure) 행이 존재하는가. 개별 구조가 도면에 그려졌는지(in_figure)와 다르다.
            # figure.csv에 행이 없으면 '도면 없음'인지 '파싱 안 됨'인지 구분할 수 없어 unknown.
            'figure_exist': TRUE if figures_by_patent.get(pid) else UNKNOWN,
            'figure_row_count': len(figures_by_patent.get(pid) or []),
            'location_status': status,
            'location_evidence': evidence,
            'source_version': src_version,
        })

    print('[5/5] 저장 및 커버리지 집계...')
    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, 'structure_location_metadata.jsonl')
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def tally(key):
        c = defaultdict(int)
        for rec in records:
            c[rec[key]] += 1
        return dict(c)

    coverage = {
        'source_version': src_version,
        'source_files': src_detail,
        'patent_molecule_rows_input': len(pm_rows),
        'duplicate_pair_rows_removed': duplicate_rows,
        'unique_patent_molecule_pairs': len(records),
        'unique_patents': len({r['patent_id'] for r in records}),
        'unique_molecules': len({r['mol_id'] for r in records}),
        'molecules_with_unparseable_smiles': sum(1 for r in records if not r['canonical_smiles']),
        'in_claim': tally('in_claim'),
        'in_figure': tally('in_figure'),
        'location_status': tally('location_status'),
        'figure_exist': tally('figure_exist'),
        'db_occurrence_count': tally('db_occurrence_count'),
        'claim_occurrence_status': tally('claim_occurrence_status'),
        'claim_occurrence_count_known': sum(1 for r in records if r['claim_occurrence_count'] is not None),
        'claim_occurrence_count_null': sum(1 for r in records if r['claim_occurrence_count'] is None),
        'claim_count_with_structure_known': sum(
            1 for r in records if r['claim_count_with_structure'] is not None),
        'records_with_unverified_figure_evidence': sum(
            1 for r in records if (r['location_evidence'] or {}).get('unverified')),
        'evidence_sources_scanned': {
            'ocsr_cache': ocsr_stats,
            'patcid_epo': patcid_stats,
            'claim_element_smiles': 'claim_element.csv 전수 스캔',
            'figure_ocsr_smiles': 'figure.csv 전수 스캔',
            'patent_molecule_location': 'patent_molecule.csv 전수 스캔',
            'claims_text_extraction': 'patent_content.claims_text 전수 스캔(InChIKey/SMILES)',
        },
        'evidence_source_availability': {
            'patents_with_claim_element_smiles_annotation': sum(1 for v in patent_has_claim_smiles_annotation.values() if v),
            'patents_with_figure_ocsr_annotation': sum(1 for v in patent_has_figure_ocsr_annotation.values() if v),
            'patent_molecule_rows_with_location': sum(1 for r in pm_rows if nonempty(r.get('location'))),
            'patent_molecule_rows_with_unit_ref': sum(1 for r in pm_rows if nonempty(r.get('unit_ref'))),
        },
    }
    out_json = os.path.join(args.out_dir, 'structure_location_coverage.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)

    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    print(f'\n저장됨: {out_jsonl}\n저장됨: {out_json}')


if __name__ == '__main__':
    main()
