import logging
import os
import tempfile
import uuid
import zipfile
from flask import send_file
import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from flask import Flask, render_template, jsonify, url_for
from huggingface_hub import hf_hub_download
from pdf2image import convert_from_path
from io import BytesIO
import re
import pytesseract
from flask import abort, request

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SEGMENTS_DIR = os.getenv('SEGMENTS_DIR', "./static/segments")
TEMP_DIR = os.getenv('TEMP_DIR', tempfile.gettempdir())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

memory_store = {}

# ---------------------------
# 1) 모델 로드 (DocLayout-YOLO)
# ---------------------------
filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(filepath)


def ocr_text_from_crop(image_rgb: np.ndarray) -> str:
    """Tesseract를 이용해 RGB 이미지 영역에서 텍스트를 추출"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    custom_oem_psm_config = r'--oem 3 --psm 4'
    text = pytesseract.image_to_string(thresh, lang='eng+kor', config=custom_oem_psm_config)
    return text.strip()


def x_overlap_filter(table_box: list[int], cap_box: list[int],
                     min_overlap_ratio: float = 0.3) -> bool:
    """
    두 바운딩박스의 x축 겹침 비율이 특정 임계값 이상인지 판단
    """
    tx1, ty1, tx2, ty2 = table_box
    cx1, cy1, cx2, cy2 = cap_box

    overlap_left = max(tx1, cx1)
    overlap_right = min(tx2, cx2)
    overlap_width = max(0, overlap_right - overlap_left)

    cap_width = cx2 - cx1
    if cap_width <= 0:
        return False

    ratio = overlap_width / float(cap_width)
    return (ratio >= min_overlap_ratio)


def find_table_caption_with_priority(
        table_box: list[int],
        table_caption_boxes: list[list[int]],
        max_distance: int = 80,
        min_overlap_ratio: float = 0.5
):
    """
    table_box 에 대해, 아래쪽→위쪽 순으로 가장 가까운 캡션 박스를 찾는다
    """
    tx1, ty1, tx2, ty2 = table_box

    bottom_candidates = []
    top_candidates = []

    for cap_box in table_caption_boxes:
        cx1, cy1, cx2, cy2 = cap_box

        # 아래쪽 후보
        if cy1 >= ty2:
            dist = cy1 - ty2
            if dist <= max_distance:
                if x_overlap_filter(table_box, cap_box, min_overlap_ratio):
                    bottom_candidates.append((cap_box, dist))

        # 위쪽 후보
        elif cy2 <= ty1:
            dist = ty1 - cy2
            if dist <= max_distance:
                if x_overlap_filter(table_box, cap_box, min_overlap_ratio):
                    top_candidates.append((cap_box, dist))

    # 거리 오름차순 정렬
    bottom_candidates.sort(key=lambda x: x[1])
    top_candidates.sort(key=lambda x: x[1])

    best_box = None
    best_distance = float('inf')
    pos = ""

    # 1) 아래쪽 후보가 있으면 가장 가까운 것
    if bottom_candidates:
        best_box, best_distance = bottom_candidates[0]
        pos = "bottom"
    else:
        # 2) 아래쪽 후보 없으면, 위쪽 중 가장 가까운 것
        if top_candidates:
            best_box, best_distance = top_candidates[0]
            pos = "top"

    return best_box, best_distance, pos


def crop_region(orig_np: np.ndarray, bbox):
    """bounding box [x1, y1, x2, y2] 해당 영역을 RGB로 잘라 반환"""
    x1, y1, x2, y2 = bbox
    return orig_np[y1:y2, x1:x2, :]


def process_pdf(pdf_path: str, extract_captions: bool = True):
    """
    PDF 파일에서 그림/표(figure, table)와 (선택적으로) 캡션을 추출
    - extract_captions=True 이면 학술논문 형태 → 이미지 + 캡션 매핑
    - extract_captions=False 이면 일반 PDF → 이미지만 추출
    """
    caption_image_dict = {}  # { caption_text: image_np }

    # PDF → PIL 이미지 목록
    pages = convert_from_path(pdf_path, 300)

    for page_idx, pil_image in enumerate(pages):
        orig_np = np.array(pil_image)  # (H,W,3) RGB

        # doclayout_yolo 추론
        det_res = model.predict(
            pil_image,
            imgsz=1024,
            conf=0.3,
            device="cuda:0",
            iou=0.6
        )
        result = det_res[0]

        figure_boxes = []
        caption_boxes = []
        table_boxes = []
        table_caption_boxes = []

        # 클래스별 분류
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            label = result.names[cls_idx]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if label == "figure":
                figure_boxes.append(xyxy)
            elif label == "figure_caption":
                caption_boxes.append(xyxy)
            elif label == "table":
                table_boxes.append(xyxy)
            elif label == "table_caption":
                table_caption_boxes.append(xyxy)

        # 1) figure 처리
        for fig_box in figure_boxes:
            fig_crop = crop_region(orig_np, fig_box)

            if extract_captions:
                # figure_caption 매핑
                best_box, dist, pos = find_table_caption_with_priority(
                    fig_box, caption_boxes
                )
                if best_box is not None:
                    caption_img = crop_region(orig_np, best_box)
                    caption_text = ocr_text_from_crop(caption_img)
                    print(f"[FIGURE] 캡션 찾음: {caption_text} (pos={pos})")
                    caption_image_dict[caption_text] = fig_crop
                else:
                    # 캡션이 없다면, 키를 빈 문자열 혹은 "Figure" + index 등으로 저장
                    caption_key = f"Figure(NoCaption)_{page_idx}"
                    caption_image_dict[caption_key] = fig_crop
            else:
                # 일반 PDF: caption 추출 X
                # 캡션 키 대신 자동으로 "Figure_<페이지>_<난수>" 로 저장 등
                caption_key = f"Figure_{page_idx}_{uuid.uuid4()}"
                caption_image_dict[caption_key] = fig_crop

        # 2) table 처리
        for tab_box in table_boxes:
            tab_crop = crop_region(orig_np, tab_box)

            if extract_captions:
                # table_caption 매핑
                best_box, dist, pos = find_table_caption_with_priority(
                    tab_box, table_caption_boxes
                )
                if best_box is not None:
                    caption_img = crop_region(orig_np, best_box)
                    caption_text = ocr_text_from_crop(caption_img)
                    print(f"[TABLE] 캡션 찾음: {caption_text} (pos={pos})")
                    caption_image_dict[caption_text] = tab_crop
                else:
                    caption_key = f"Table(NoCaption)_{page_idx}"
                    caption_image_dict[caption_key] = tab_crop
            else:
                # 일반 PDF: caption 추출 X
                caption_key = f"Table_{page_idx}_{uuid.uuid4()}"
                caption_image_dict[caption_key] = tab_crop

    return caption_image_dict


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('mainWeb.html')


@app.route('/download/<filename>')
def download_from_memory(filename):
    """
    메모리에 저장된 이미지를 다운로드(또는 브라우저 표시)
    """
    if filename not in memory_store:
        abort(404, description="해당 ID에 해당하는 파일이 없습니다.")

    img_bytes = memory_store[filename]
    memfile = BytesIO(img_bytes)
    memfile.seek(0)
    return send_file(
        memfile,
        mimetype='image/png',
        as_attachment=True,
        download_name=f"{filename}.png"
    )


@app.route('/download-zip/<filename>')
def download_zip(filename: str):
    try:
        base_name = filename.split('_')[0]
        logger.info(f"[download_zip] ZIP 다운로드 요청: base_name={base_name}")

        zip_buffer = BytesIO()
        pattern = re.compile(rf"^{re.escape(base_name)}_IMG_[0-9a-fA-F-]+$")
        matched = False

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_id, img_bytes in memory_store.items():
                if pattern.match(file_id):
                    matched = True
                    arcname = f"{file_id}.png"
                    zip_file.writestr(arcname, img_bytes)

        if not matched:
            return jsonify({'error': '해당 베이스 이름의 파일이 없습니다.'}), 404

        zip_buffer.seek(0)
        zip_filename = f"{base_name}_images.zip"

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )

    except Exception as e:
        logger.error(f"[download_zip] ZIP 파일 생성 중 에러: {e}")
        abort(500, description="내부 서버 오류입니다.")


# ---------------------------------------------------------------------------
# “일반 PDF” 업로드: 이미지만 추출
# ---------------------------------------------------------------------------
@app.route('/upload_normal', methods=['POST'])
def upload_normal_pdf():
    """
    일반 PDF 업로드 핸들러
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400

        pdf_file = request.files['file']
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': '유효하지 않은 파일 형식입니다. PDF를 업로드하세요.'}), 400

        # 임시 PDF 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as tmp_pdf:
            pdf_file.save(tmp_pdf.name)
            temp_pdf_path = tmp_pdf.name

        # PDF 처리 (캡션 추출=False)
        extracted_dict = process_pdf(temp_pdf_path, extract_captions=False)

        # 임시 PDF 삭제
        os.remove(temp_pdf_path)

        # 이미지 결과를 memory_store에 저장 → JSON 응답
        base_filename = os.path.splitext(pdf_file.filename)[0]
        output_data = []

        for caption, img_rgb in extracted_dict.items():
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            success, encoded_img = cv2.imencode(".png", img_bgr)
            if success:
                img_bytes = encoded_img.tobytes()
                file_id = base_filename + '_IMG_' + str(uuid.uuid4())
                memory_store[file_id] = img_bytes

                download_url = url_for('download_from_memory', filename=file_id, _external=True)
                output_data.append({
                    'url': download_url,
                    'caption': caption
                })

        response_data = {
            'images': output_data,
            'uploaded_filename': base_filename
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"[upload_normal_pdf] 에러: {e}")
        return jsonify({'error': '내부 서버 오류입니다.'}), 500


# ---------------------------------------------------------------------------
# “학술 논문 PDF” 업로드: 이미지 + 캡션 매핑
# ---------------------------------------------------------------------------
@app.route('/upload_academic', methods=['POST'])
def upload_academic_pdf():
    """
    학술 논문 PDF 업로드 핸들러
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400

        pdf_file = request.files['file']
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': '유효하지 않은 파일 형식입니다. PDF를 업로드하세요.'}), 400

        # 임시 PDF 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as tmp_pdf:
            pdf_file.save(tmp_pdf.name)
            temp_pdf_path = tmp_pdf.name

        # PDF 처리 (캡션 추출=True)
        extracted_dict = process_pdf(temp_pdf_path, extract_captions=True)

        # 임시 PDF 삭제
        os.remove(temp_pdf_path)

        # 이미지 결과 memory_store에 저장 → JSON 응답
        base_filename = os.path.splitext(pdf_file.filename)[0]
        output_data = []

        for caption, img_rgb in extracted_dict.items():
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            success, encoded_img = cv2.imencode(".png", img_bgr)
            if success:
                img_bytes = encoded_img.tobytes()
                file_id = base_filename + '_IMG_' + str(uuid.uuid4())
                memory_store[file_id] = img_bytes

                download_url = url_for('download_from_memory', filename=file_id, _external=True)
                output_data.append({
                    'url': download_url,
                    'caption': caption
                })

        response_data = {
            'images': output_data,
            'uploaded_filename': base_filename
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"[upload_academic_pdf] 에러: {e}")
        return jsonify({'error': '내부 서버 오류입니다.'}), 500


# -----------------------------------------------------------------------------
# 6. Flask 앱 실행
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # debug=True로 두면, 수정 후 서버 자동 리로드
    app.run(debug=True, host='0.0.0.0', port=5000)
