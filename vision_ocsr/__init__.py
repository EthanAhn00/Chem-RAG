# Import submodules separately with try/except so a missing optional dependency
# (e.g. cv2/DECIMER) does not prevent using other components such as OPS access.
from .epo_ops_fetcher import EPOOPSFetcher

try:
    from .chem_vision_cropper import ChemVisionCropper
except ImportError:
    ChemVisionCropper = None

try:
    from .ocsr_smiles_converter import OCSRSMILESConverter
except ImportError:
    OCSRSMILESConverter = None
