import re

class ChemicalTokenizer:
    """화학 명명법, IUPAC 용어, SMILES 표기법을 위한 특화 토크나이저"""

    def __init__(self):
        # 화학식, 치환기, 영숫자를 매칭하는 정규표현식
        self.chem_pattern = re.compile(r'\b[A-Za-z0-9\-\(\)\[\]\,\=\+]{2,}\b')

    def tokenize(self, text):
        # 화학 청구항 텍스트를 정규화된 화학 용어 토큰으로 분리
        if not text:
            return []
        text = text.lower()
        tokens = self.chem_pattern.findall(text)
        return [t for t in tokens if len(t) > 1]
