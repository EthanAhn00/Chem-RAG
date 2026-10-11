"""구조(특허-분자) 컬렉션 접근 계층 + RDKit exact Tanimoto 계산.

Qdrant에 저장된 Morgan 지문 sparse 벡터는 '후보 생성'에만 쓰고, 유사도 값 자체는
반드시 RDKit으로 다시 계산한다(Qdrant의 dot/cosine != Tanimoto).
"""
from collections import defaultdict

from qdrant_client import models

from .schemas import UNKNOWN


def morgan_bits(smiles, radius=2, nbits=2048):
    """SMILES -> Morgan(ECFP4) on-bit 인덱스 리스트. 파싱 실패 시 None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return list(fp.GetOnBits())


def build_fingerprint(smiles, radius=2, nbits=2048):
    """RDKit ExplicitBitVect 반환(정확한 Tanimoto 계산용). 실패 시 None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def max_tanimoto(fps_a, fps_b):
    """두 특허의 분자 집합 간 최대 Tanimoto. 한쪽이라도 비면 None(0이 아님)."""
    if not fps_a or not fps_b:
        return None
    from rdkit import DataStructs
    best = 0.0
    for fa in fps_a:
        for fb in fps_b:
            s = DataStructs.TanimotoSimilarity(fa, fb)
            if s > best:
                best = s
    return round(best, 4)


class StructureRepository:
    """구조 컬렉션 조회 + 특허별 지문 캐시."""

    def __init__(self, client, settings, collection=None):
        self.client = client
        self.settings = settings
        self.collection = collection or settings.structures_collection
        self._fp_cache = {}

    # ------------------------------------------------------------ 조회
    def iter_all(self, with_vectors=False, batch=512):
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=batch, offset=offset,
                with_payload=True, with_vectors=with_vectors)
            for p in points:
                yield p
            if offset is None:
                break

    def structures_for_patent(self, patent_id):
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key='patent_id', match=models.MatchValue(value=patent_id))]),
            limit=1000, with_payload=True, with_vectors=False)
        return [p.payload for p in points]

    def load_smiles_index(self):
        """patent_id -> [smiles] 전체 로드(평가 시 한 번만 호출)."""
        index = defaultdict(list)
        for p in self.iter_all():
            payload = p.payload or {}
            smi = payload.get('smiles')
            if smi:
                index[payload.get('patent_id')].append(smi)
        return dict(index)

    # ------------------------------------------------------- 지문/유사도
    def fingerprints_for_patent(self, patent_id, smiles_index=None):
        if patent_id in self._fp_cache:
            return self._fp_cache[patent_id]
        smiles_list = (smiles_index or {}).get(patent_id)
        if smiles_list is None:
            smiles_list = [s.get('smiles') for s in self.structures_for_patent(patent_id)]
        fps = []
        for smi in smiles_list or []:
            fp = build_fingerprint(smi, self.settings.morgan_radius, self.settings.morgan_nbits)
            if fp is not None:
                fps.append(fp)
        self._fp_cache[patent_id] = fps
        return fps

    def exact_tanimoto(self, patent_a, patent_b, smiles_index=None):
        """두 특허 간 exact Tanimoto(최댓값). 어느 쪽이든 구조가 없으면 None."""
        return max_tanimoto(
            self.fingerprints_for_patent(patent_a, smiles_index),
            self.fingerprints_for_patent(patent_b, smiles_index))

    def approximate_candidates(self, smiles, limit=50):
        """지문 sparse 벡터로 유사 구조 후보를 뽑는다(근사).

        반환 점수는 Tanimoto가 아니라 on-bit 내적이다. 정렬/필터 용도로만 쓰고
        실제 유사도는 exact_tanimoto()로 다시 계산해야 한다.
        """
        bits = morgan_bits(smiles, self.settings.morgan_radius, self.settings.morgan_nbits)
        if not bits:
            return []
        res = self.client.query_points(
            self.collection,
            query=models.SparseVector(indices=bits, values=[1.0] * len(bits)),
            using=self.settings.fingerprint_vector_name, limit=limit, with_payload=True)
        return [{'patent_id': p.payload.get('patent_id'), 'mol_id': p.payload.get('mol_id'),
                 'smiles': p.payload.get('smiles'),
                 'approx_overlap_score': float(p.score), 'is_tanimoto': False} for p in res.points]

    # ------------------------------------------------------------ 통계
    def location_coverage_stats(self):
        counts = {'in_claim': defaultdict(int), 'in_figure': defaultdict(int),
                  'location_status': defaultdict(int), 'claim_occurrence_count': {'known': 0, 'null': 0},
                  'total_points': 0}
        for p in self.iter_all():
            payload = p.payload or {}
            counts['total_points'] += 1
            counts['in_claim'][payload.get('in_claim', UNKNOWN)] += 1
            counts['in_figure'][payload.get('in_figure', UNKNOWN)] += 1
            counts['location_status'][payload.get('location_status', 'unset')] += 1
            if payload.get('claim_occurrence_count') is None:
                counts['claim_occurrence_count']['null'] += 1
            else:
                counts['claim_occurrence_count']['known'] += 1
        for k in ('in_claim', 'in_figure', 'location_status'):
            counts[k] = dict(counts[k])
        return counts
