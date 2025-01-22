import logging
import os
import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from pdf2image import convert_from_path

# (추가) pytesseract 설치 및 경로 설정 (windows 설치 경로 예시)
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------------------------------------------------------
# Logging 설정
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ocr_text_from_crop(image_rgb: np.ndarray) -> str:
    """
    Tesseract를 이용해 RGB 이미지 영역에서 텍스트를 추출.
    """
    # OpenCV용 BGR이 아니라, 이미 RGB임을 가정
    # 그레이 변환 → Threshold or Adaptive → OCR
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # 간단히 이진화 (더 정교한 전처리를 해도 됨)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang='eng+kor')  # 필요한 언어 지정
    return text.strip()


def find_closest_caption(fig_box, caption_boxes):
    """
    하나의 figure 박스와 여러 figure_caption 박스들 중
    가장 가까운(세로거리 기준) 캡션 박스를 찾아 반환.

    - fig_box: [x1, y1, x2, y2]
    - caption_boxes: List of [x1, y1, x2, y2]

    여기서는 간단히 figure 하단( y2 )과 caption 상단( y1 )의 차이를 기준으로 최소값 찾기.
    더 정교하게는 실제로 '상단/하단' 방향에 있는지, 거리 제한이 있는지 등을 고려 가능.
    """
    fx1, fy1, fx2, fy2 = fig_box
    best_box = None
    best_distance = float('inf')

    for cap_box in caption_boxes:
        cx1, cy1, cx2, cy2 = cap_box
        distance = cy1 - fy2  # figure 하단 y2와 caption 상단 y1의 차이
        if distance < 0:
            # 캡션이 figure 위에 있을 수도 있으니,
            # "figure 위쪽에도 캡션이 있을 수 있다"면 다른 계산( fy1 - cy2 )도 고려
            continue
        if 0 <= distance < best_distance:
            best_distance = distance
            best_box = cap_box

    return best_box


def crop_region(orig_np: np.ndarray, bbox):
    """
    bounding box [x1, y1, x2, y2]에 해당하는 영역을 잘라서 return (RGB)
    orig_np는 PIL에서 np.array 변환한 (H,W,3) RGB
    """
    x1, y1, x2, y2 = bbox
    return orig_np[y1:y2, x1:x2, :]


def main():
    # 1) 모델 로드 (DocLayout-YOLO)
    filepath = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    model = YOLOv10(filepath)

    # PDF -> Image
    pdf_file = './ATN0050547548.pdf'
    pages = convert_from_path(pdf_file, 300)

    # 결과 저장 폴더
    save_dir = "./extracted_figures"
    os.makedirs(save_dir, exist_ok=True)

    for page_idx, pil_image in enumerate(pages):
        # PIL -> numpy (RGB)
        orig_np = np.array(pil_image)

        # 2) doclayout_yolo 추론
        det_res = model.predict(
            pil_image,
            imgsz=1024,
            conf=0.2,
            device="cuda:0"  # or 'cpu'
        )
        result = det_res[0]

        # 3) figure, figure_caption 박스 나눠서 담기
        figure_boxes = []
        caption_boxes = []
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            label = result.names[cls_idx]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            # xyxy = [x1, y1, x2, y2]

            if label == "figure":
                figure_boxes.append(xyxy)
            elif label == "figure_caption":
                caption_boxes.append(xyxy)

        # 4) 매핑: figure - figure_caption
        #    간단히 "figure 아래쪽 중 가장 가까운 caption"을 찾는 로직 예시
        for i, fig_box in enumerate(figure_boxes):
            matched_cap_box = find_closest_caption(fig_box, caption_boxes)

            # figure 이미지 crop
            fig_crop = crop_region(orig_np, fig_box)
            # figure_caption 이미지 crop + OCR
            if matched_cap_box is not None:
                cap_crop = crop_region(orig_np, matched_cap_box)
                caption_text = ocr_text_from_crop(cap_crop)
            else:
                caption_text = ""  # 못 찾았으면 빈 문자열

            # 5) figure 이미지 저장 + 캡션 텍스트 기록
            out_filename = os.path.join(
                save_dir, f"page{page_idx}_fig{i}.jpg"
            )
            fig_bgr = cv2.cvtColor(fig_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_filename, fig_bgr)

            print(f"[PAGE {page_idx}] figure {i} 저장: {out_filename}")
            print(f"  → caption: {caption_text}\n")

            # (선택) 캡션 텍스트를 별도 txt나 JSON으로 저장할 수도 있음
            # 예시: 같은 이름의 .txt 파일로 저장
            if caption_text:
                txt_filename = os.path.splitext(out_filename)[0] + ".txt"
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(caption_text)

