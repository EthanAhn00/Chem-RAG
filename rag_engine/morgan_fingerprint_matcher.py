from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# Suppress RDKit C++ parser warnings for invalid or empty SMILES.
RDLogger.DisableLog('rdApp.*')


class MorganFingerprintMatcher:
    """Chemical-structure similarity using Morgan 2048-bit fingerprints and Tanimoto.

    RDKit's deterministic algorithm requires no training: atom neighborhoods of radius 2 (ECFP4) are hashed into a 2048-bit vector and compared by Tanimoto.
    Accepts actual SMILES only. Return None if either input cannot be parsed as a molecule; never treat claim text as a substitute SMILES.
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
