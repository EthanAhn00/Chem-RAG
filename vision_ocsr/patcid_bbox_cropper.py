import os
from PIL import Image

# 현재 비활성 경로 - OCSRPipeline.ENABLE_BBOX_CROP_FALLBACK=False라 호출되지 않음


def crop_by_patcid_bbox(image_path, bbox_inches, page_width_in, page_height_in, output_path, padding_frac=0.08):
    """PatCID가 이미 계산해둔 bbox(인치 단위, [x0,y0,x1,y1])로 화학구조 영역을 정확히 크롭

    범용 윤곽선 검출(chem_vision_cropper.py)과 달리, IBM PatCID의 학습된 검출기가
    실제 이 페이지의 이 위치에 화학구조가 있다고 판정한 좌표를 그대로 사용하므로
    텍스트/화살표/도형 등 엉뚱한 영역을 잘라내는 문제가 없음

    반환: 성공 시 output_path, 실패 시 None
    """
    if not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size

        # 인치 -> 픽셀 배율(해당 페이지의 실제 dpi). x/y를 각각 계산해 종횡비가
        # 다르게 스캔된 경우도 대응
        dpi_x = img_w / page_width_in if page_width_in else img_w / 8.5
        dpi_y = img_h / page_height_in if page_height_in else img_h / 11.0

        x0, y0, x1, y1 = bbox_inches
        px0, py0, px1, py1 = x0 * dpi_x, y0 * dpi_y, x1 * dpi_x, y1 * dpi_y

        # bbox가 다소 타이트할 수 있어 여백을 조금 더해줌(비율 기반 padding)
        w, h = px1 - px0, py1 - py0
        pad_x, pad_y = w * padding_frac, h * padding_frac
        left = max(0, int(px0 - pad_x))
        top = max(0, int(py0 - pad_y))
        right = min(img_w, int(px1 + pad_x))
        bottom = min(img_h, int(py1 + pad_y))

        if right <= left or bottom <= top:
            return None

        crop = img.crop((left, top, right, bottom))
        crop.save(output_path)
        return output_path
    except Exception:
        return None
