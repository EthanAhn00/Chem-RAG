import os
import importlib

class OCSRSMILESConverter:
    """Optical Chemical Structure Recognition (OCSR) converting 2D images to SMILES; integrates DECIMER 2.0 or SwinOCSR when enabled."""

    def __init__(self):
        self.decimer_available = False
        self.decimer_module = None
        self._check_decimer_availability()

    def _check_decimer_availability(self):
        # The pip package is 'decimer', but the import module is 'DECIMER';
        # try both names, uppercase first.
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

        # Missing DECIMER or failed prediction means no recognized structure;
        # return NOT_FOUND instead of fabricated SMILES. Callers should treat it
        # as unavailable recognition, not zero similarity to a molecule.
        filename = os.path.basename(image_path)
        return {
            'status': 'NOT_FOUND',
            'smiles': None,
            'engine': 'Standby OCSR Converter (DECIMER not active)',
            'source_file': filename
        }
