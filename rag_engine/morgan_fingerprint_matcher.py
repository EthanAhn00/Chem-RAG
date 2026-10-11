from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# 유효하지 않거나 빈 SMILES 문자열에 대한 RDKit C++ 파서 경고 억제
RDLogger.DisableLog('rdApp.*')


class MorganFingerprintMatcher:
    """Morgan 2048-bit Fingerprint 기반 화학구조 Tanimoto 유사도 매처

    RDKit의 결정론적(deterministic) 화학정보학 알고리즘(학습 없음). 분자를 원자
    주변 반경 2(ECFP4)로 순회해 2048비트 벡터로 해시한 뒤 Tanimoto 계수를 계산

    실제 SMILES 문자열만 입력으로 받으며, 두 입력 중 하나라도 유효한 분자로
    파싱되지 않으면 None을 반환(청구항 텍스트를 SMILES로 대신 취급하는 폴백 없음)
    """

    def __init__(self):
        self.rdkit_available = True

    def compute_tanimoto_similarity(self, smiles_q, smiles_d):
        if not smiles_q or not smiles_d:
            return None
        try:
            mol_q = Chem.MolFromSmiles(smiles_q)
            mol_d = Chem.MolFromSmiles(smiles_d)
            if mol_q is None or mol_d is None:
                return None
            fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, 2, nBits=2048)
            fp_d = AllChem.GetMorganFingerprintAsBitVect(mol_d, 2, nBits=2048)
            return round(DataStructs.TanimotoSimilarity(fp_q, fp_d), 4)
        except Exception:
            return None
