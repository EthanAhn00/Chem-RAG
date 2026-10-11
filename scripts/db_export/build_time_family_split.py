import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR, ID_PATENT_MAP_PATH


class UnionFind:
    """문자열 키(특허번호)에 대한 간단한 disjoint-set(union-find)"""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_id_patent_map(path):
    id_map = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            try:
                rid = int(row['id'])
            except (KeyError, ValueError, TypeError):
                continue
            year = row.get('file_year')
            id_map[rid] = {
                'target_patent_id': row.get('T_id') or None,
                'prior_patent_id': row.get('P_id') or None,
                'file_year': int(float(year)) if year not in (None, '', 'nan') else None,
            }
    return id_map


def build_split(dataset_path, id_patent_map_path, test_fraction=0.2, out_path=None, summary_path=None):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    id_map = load_id_patent_map(id_patent_map_path)

    # 각 레코드에 특허번호/연도 정보 부여
    enriched = []
    for r in records:
        stem = r['filename'].replace('.json', '')
        link = id_map.get(int(stem)) if stem.isdigit() else None
        enriched.append({
            'filename': r['filename'],
            'target_patent_id': (link or {}).get('target_patent_id'),
            'prior_patent_id': (link or {}).get('prior_patent_id'),
            'file_year': (link or {}).get('file_year'),
        })

    resolved = [r for r in enriched if r['target_patent_id'] and r['file_year']]
    unresolved = [r for r in enriched if not (r['target_patent_id'] and r['file_year'])]

    # Cutoff 연도: 특허 패밀리 병합 이전, resolved 레코드들의 연도 분포에서
    # (1 - test_fraction) 분위수 - "가장 최근 test_fraction 비율을 TEST로" 하는 초기 split
    years_sorted = sorted(r['file_year'] for r in resolved)
    if years_sorted:
        idx = int(len(years_sorted) * (1 - test_fraction))
        idx = min(idx, len(years_sorted) - 1)
        cutoff_year = years_sorted[idx]
    else:
        cutoff_year = None

    # 특허번호 기준 union-find: target/prior 특허번호를 다른 레코드와 공유하면
    # 반드시 같은 split에 속하도록 함(패밀리/자기 누출 방지)
    uf = UnionFind()
    for r in resolved:
        keys = [k for k in (r['target_patent_id'], r['prior_patent_id']) if k]
        for k in keys:
            uf.find(k)
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)

    # 서로 다른 레코드끼리 특허번호(target 또는 prior) 중 하나라도 공유하면 함께 묶음
    key_to_root = {}
    for r in resolved:
        for k in (r['target_patent_id'], r['prior_patent_id']):
            if not k:
                continue
            root = uf.find(k)
            if k in key_to_root and key_to_root[k] != root:
                uf.union(key_to_root[k], root)
            key_to_root[k] = uf.find(k)

    # 컴포넌트 -> 해당 컴포넌트에서 관측된 최소 연도 (보수적 접근: 패밀리 내 어디서든
    # cutoff 이전 등장이 있으면 패밀리 전체를 TRAIN으로 배정)
    component_min_year = {}
    record_component = {}
    for r in resolved:
        keys = [k for k in (r['target_patent_id'], r['prior_patent_id']) if k]
        comp = uf.find(keys[0]) if keys else None
        record_component[r['filename']] = comp
        if comp is not None:
            component_min_year[comp] = min(component_min_year.get(comp, r['file_year']), r['file_year'])

    split = {}
    for r in resolved:
        comp = record_component[r['filename']]
        comp_min_year = component_min_year.get(comp, r['file_year'])
        if cutoff_year is not None and comp_min_year >= cutoff_year:
            split[r['filename']] = 'test'
        else:
            split[r['filename']] = 'train'

    for r in unresolved:
        split[r['filename']] = 'train_unresolved'  # 특허번호/연도 없음 -> leak-free 검증 불가, test에서 제외

    counts = {
        'train': sum(1 for v in split.values() if v == 'train'),
        'test': sum(1 for v in split.values() if v == 'test'),
        'train_unresolved': sum(1 for v in split.values() if v == 'train_unresolved'),
    }

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(split, f, indent=2, ensure_ascii=False)

    summary = {
        'total_records': len(records),
        'resolved_with_year': len(resolved),
        'unresolved': len(unresolved),
        'cutoff_year': cutoff_year,
        'target_test_fraction': test_fraction,
        'actual_counts': counts,
        'actual_test_fraction_of_resolved': round(counts['test'] / len(resolved), 4) if resolved else None,
        'note': 'train_unresolved = records without a recovered patent number/year; kept out of test because leakage cannot be verified for them. test = connected component (by shared target/prior patent number) whose earliest known appearance is >= cutoff_year.',
    }
    if summary_path:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return split, summary


if __name__ == '__main__':
    dataset_path = os.path.join(DATA_DIR, 'reference', 'epo_section_c_benchmark_dataset.json')
    out_path = os.path.join(DATA_DIR, 'reference', 'train_test_split.json')
    summary_path = os.path.join(DATA_DIR, 'reference', 'train_test_split_summary.json')
    split, summary = build_split(dataset_path, ID_PATENT_MAP_PATH, test_fraction=0.2, out_path=out_path, summary_path=summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
