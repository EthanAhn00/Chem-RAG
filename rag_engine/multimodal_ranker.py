from .hybrid_retriever import HybridChemicalRetriever
from .morgan_fingerprint_matcher import MorganFingerprintMatcher


class MultimodalChemicalRanker:
    """텍스트, Morgan Tanimoto Fingerprint, 도면(Vision) 신호를 결합하는 동적 멀티모달 랭커

    학습된 신경망 모델이 아니라 세 독립 신호(text_score/tanimoto_score/vision_bonus)를
    점수 레벨에서 선형 결합(late fusion)하는 방식:
      Score(Q, D) = alpha*TextSim + gamma*TanimotoSim + beta*VisionBonus, alpha+gamma+beta = 1.0
    (기본값 alpha/gamma/beta = 0.5/0.3/0.2)

    Tanimoto/Vision 항은 실제 데이터가 뒷받침될 때만 포함되며, 없는 항은 조용히 0점
    처리하지 않고 사용 가능한 항들만으로 가중치를 재정규화(합이 다시 1.0이 되도록)
    - 그렇지 않으면 구조/도면 데이터가 없는 레코드가 부당하게 낮은 점수를 받게 됨
    """

    def __init__(self, text_weight=0.5, tanimoto_weight=0.3, vision_weight=0.2):
        self.text_weight = text_weight
        self.tanimoto_weight = tanimoto_weight
        self.vision_weight = vision_weight
        self.retriever = HybridChemicalRetriever()
        self.fp_matcher = MorganFingerprintMatcher()

    def rank_patent_pair(self, target_claim, prior_claim, target_smiles=None, prior_smiles=None,
                          has_drawing=False, drawing_count=0):
        # 1. 텍스트 유사도 - 항상 계산 가능
        text_score = self.retriever.compute_text_similarity(target_claim, prior_claim)

        # 2. 화학구조 Tanimoto 유사도 - 두 SMILES가 모두 유효한 분자로 해소된 경우에만 계산
        # None = "구조 데이터 없음"(유사도 0과는 구분)
        tanimoto_score = self.fp_matcher.compute_tanimoto_similarity(target_smiles, prior_smiles)
        has_structure = tanimoto_score is not None

        # 3. Vision 보너스 - 매칭된 도면이 있을 때만 적용되는 존재/개수 기반 휴리스틱
        # (도면 내용을 분석해 유사도를 매기는 것이 아님)
        has_vision = bool(has_drawing) and drawing_count > 0
        vision_bonus = min(1.0, 0.5 + 0.1 * drawing_count) if has_vision else 0.0

        components = [('text', self.text_weight, text_score)]
        if has_structure:
            components.append(('tanimoto', self.tanimoto_weight, tanimoto_score))
        if has_vision:
            components.append(('vision', self.vision_weight, vision_bonus))

        # 사용 가능한 항들의 가중치 합으로 나누어 재정규화
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
