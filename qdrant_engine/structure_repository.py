"""Structure (patent-molecule) collection access and RDKit exact Tanimoto. Qdrant Morgan sparse vectors are for candidate generation only; recompute similarity with RDKit because Qdrant dot/cosine is not Tanimoto."""
from collections import defaultdict

from qdrant_client import models

from .schemas import UNKNOWN


def morgan_bits(smiles, radius=2, nbits=2048):
    """SMILES -> Morgan (ECFP4) on-bit indexes; None on parsing failure."""
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
    """Return RDKit ExplicitBitVect for exact Tanimoto, or None on failure."""
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
    """Maximum Tanimoto across two patent molecule sets; None, not 0, if either is empty."""
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
    """Structure collection lookup and per-patent fingerprint cache."""

    def __init__(self, client, settings, collection=None):
        self.client = client
        self.settings = settings
        self.collection = collection or settings.structures_collection
        self._fp_cache = {}

    # ------------------------------------------------------------ Lookup
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
        """Load all patent_id -> [smiles] mappings, once per evaluation."""
        index = defaultdict(list)
        for p in self.iter_all():
            payload = p.payload or {}
            smi = payload.get('smiles')
            if smi:
                index[payload.get('patent_id')].append(smi)
        return dict(index)

    # ------------------------------------------------------- Fingerprints/similarity
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
        """Maximum exact Tanimoto between two patents; None if either lacks structures."""
        return max_tanimoto(
            self.fingerprints_for_patent(patent_a, smiles_index),
            self.fingerprints_for_patent(patent_b, smiles_index))

    def approximate_candidates(self, smiles, limit=50):
        """Approximate structure candidate generation using sparse fingerprints. Returned scores are on-bit dot products, not Tanimoto; use only for ranking/filtering and recompute similarity with exact_tanimoto()."""
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

    # ------------------------------------------------------------ Statistics
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
