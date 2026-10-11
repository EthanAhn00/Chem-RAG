import os
import cv2

class ChemVisionCropper:
    """도면 이미지에서 2D 화학구조 영역을 검출해 자동으로 BBox를 크롭하는 클래스"""

    def __init__(self, crop_output_dir=None):
        # 크롭 결과 저장 디렉토리 초기화
        if crop_output_dir is None:
            crop_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'crops')
        self.crop_output_dir = crop_output_dir
        os.makedirs(self.crop_output_dir, exist_ok=True)

    def crop_chemical_structure(self, image_path, patent_id):
        # 도면 이미지에서 2D 화학구조 bounding box를 크롭해 PNG로 저장
        if not os.path.exists(image_path):
            return []

        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cropped_paths = []
        crop_idx = 1

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > 120 and ch > 120 and cw < int(w * 0.9) and ch < int(h * 0.9):
                roi = img[y:y+ch, x:x+cw]
                out_name = f'{patent_id}_crop_{crop_idx}.png'
                out_path = os.path.join(self.crop_output_dir, out_name)
                cv2.imwrite(out_path, roi)
                cropped_paths.append(out_path)
                crop_idx += 1

        return cropped_paths
