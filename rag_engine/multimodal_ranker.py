from .hybrid_retriever import HybridChemicalRetriever
from .morgan_fingerprint_matcher import MorganFingerprintMatcher


class MultimodalChemicalRanker:
    """Dynamic multimodal ranker combining text, Morgan Tanimoto, and drawing signals.

    Uses linear score-level late fusion, not a learned neural network:
    Score(Q, D) = alpha*TextSim + gamma*TanimotoSim + beta*VisionBonus, alpha+gamma+beta = 1.0.
    Defaults: alpha/gamma/beta = 0.5/0.3/0.2.
    Include Tanimoto/Vision only when supported by actual data. Renormalize weights over available components to sum to 1.0 instead of silently assigning missing components zero; otherwise records without structures/drawings would be unfairly penalized.
    """

    def __init__(self, text_weight=0.5, tanimoto_weight=0.3, vision_weight=0.2):
        self.text_weight = text_weight
        self.tanimoto_weight = tanimoto_weight
        self.vision_weight = vision_weight
        self.retriever = HybridChemicalRetriever()
        self.fp_matcher = MorganFingerprintMatcher()

    def rank_patent_pair(self, target_claim, prior_claim, target_smiles=None, prior_smiles=None,
                          has_drawing=False, drawing_count=0):
        # 1. Text similarity is always computable.
        text_score = self.retriever.compute_text_similarity(target_claim, prior_claim)

        # 2. Compute Tanimoto only if both SMILES parse as valid molecules.
        # None means missing structure data, not zero similarity.
        tanimoto_score = self.fp_matcher.compute_tanimoto_similarity(target_smiles, prior_smiles)
        has_structure = tanimoto_score is not None

        # 3. Vision bonus is a presence/count heuristic requiring matched drawings.
        # It does not compare drawing contents.
        has_vision = bool(has_drawing) and drawing_count > 0
        vision_bonus = min(1.0, 0.5 + 0.1 * drawing_count) if has_vision else 0.0

        components = [('text', self.text_weight, text_score)]
        if has_structure:
            components.append(('tanimoto', self.tanimoto_weight, tanimoto_score))
        if has_vision:
            components.append(('vision', self.vision_weight, vision_bonus))

        # Renormalize by the sum of weights of available components.
        weight_sum = sum(w for _, w, _ in components)
        hybrid_score = sum(w * s for _, w, s in components) / weight_sum if weight_sum > 0 else 0.0

        return {
            'text_score': round(text_score, 4),
            'tanimoto_score': round(tanimoto_score, 4) if tanimoto_score is not None else None,
            'has_structure': has_structure,
            'has_drawing': has_vision,
            'drawing_count': drawing_count,
            'vision_bonus': round(vision_bonus, 4),
            'components_used': [name for name, _, _ in components],
            'final_hybrid_score': round(hybrid_score, 4)
        }
