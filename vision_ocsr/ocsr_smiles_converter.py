import os
import importlib

class OCSRSMILESConverter:
    """2D 이미지로부터 SMILES를 예측하는 OCSR(Optical Chemical Structure Recognition) 변환기
    활성화 시 DECIMER 2.0 또는 SwinOCSR 모델과 연동"""

    def __init__(self):
        self.decimer_available = False
        self.decimer_module = None
        self._check_decimer_availability()

    def _check_decimer_availability(self):
        # pip 배포명은 'decimer'(소문자)지만 실제 import 모듈명은 'DECIMER'(대문자) -
        # 혼동하기 쉬워 둘 다 시도(대문자 우선)
        for module_name in ('DECIMER', 'decimer'):
            try:
                self.decimer_module = importlib.import_module(module_name)
                self.decimer_available = True
                return
            except (ImportError, ModuleNotFoundError):
                continue
        self.decimer_available = False

    def convert_image_to_smiles(self, image_path):
        if not os.path.exists(image_path):
            return {'status': 'FILE_NOT_FOUND', 'smiles': None}

        if self.decimer_available and self.decimer_module:
            try:
                predict_fn = getattr(self.decimer_module, 'predict_SMILES', None)
                if predict_fn:
                    smiles = predict_fn(image_path)
                    return {'status': 'SUCCESS', 'smiles': smiles, 'engine': 'DECIMER 2.0'}
            except Exception:
                pass

        # DECIMER 미설치 또는 예측 실패 - 실제 구조를 인식하지 못했으므로 임의의
        # SMILES를 반환하지 않고 NOT_FOUND로 명시 (호출하는 쪽은 이를 "구조 인식 불가"로
        # 처리할 것 - 특정 분자와의 유사도 0과는 다름)
        filename = os.path.basename(image_path)
        return {
            'status': 'NOT_FOUND',
            'smiles': None,
            'engine': 'Standby OCSR Converter (DECIMER not active)',
            'source_file': filename
        }
