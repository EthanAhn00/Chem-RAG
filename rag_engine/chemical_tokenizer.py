import re

class ChemicalTokenizer:
    """Tokenizer specialized for chemical nomenclature, IUPAC terms, and SMILES notation."""

    def __init__(self):
        # Match chemical formulas, substituents, and alphanumeric terms.
        self.chem_pattern = re.compile(r'\b[A-Za-z0-9\-\(\)\[\]\,\=\+]{2,}\b')

    def tokenize(self, text):
        # Split chemical claims into normalized chemical-term tokens.
        if not text:
            return []
        text = text.lower()
        tokens = self.chem_pattern.findall(text)
        return [t for t in tokens if len(t) > 1]
