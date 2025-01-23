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
from flask import abort

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SEGMENTS_DIR = os.getenv('SEGMENTS_DIR', "./static/segments")
TEMP_DIR = os.getenv('TEMP_DIR', tempfile.gettempdir())

# -----------------------------------------------------------------------------
# Logging 설정
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

memory_store = {}
# 1) 모델 로드 (DocLayout-YOLO)
filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(filepath)


def ocr_text_from_crop(image_rgb: np.ndarray) -> str:
    """
    Tesseract를 이용해 RGB 이미지 영역에서 텍스트를 추출.
    """
    # OpenCV용 BGR이 아니라, 이미 RGB임을 가정
    # 그레이 변환 → Threshold or Adaptive → OCR
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    custom_oem_psm_config = r'--oem 3 --psm 4'
    text = pytesseract.image_to_string(thresh, lang='eng+kor', config=custom_oem_psm_config)
    return text.strip()


def x_overlap_filter(
        table_box: list[int],
        cap_box: list[int],
        min_overlap_ratio: float = 0.3
) -> bool:
    """
    table_box, cap_box의 x축 겹치는 구간이
    cap_box 폭의 min_overlap_ratio(기본 30%) 이상인지를 체크.

    - table_box: [tx1, ty1, tx2, ty2]
    - cap_box: [cx1, cy1, cx2, cy2]
    - overlap_ratio = overlap_width / cap_width
    """
    tx1, ty1, tx2, ty2 = table_box
    cx1, cy1, cx2, cy2 = cap_box

    overlap_left = max(tx1, cx1)
    overlap_right = min(tx2, cx2)
    overlap_width = max(0, overlap_right - overlap_left)

    cap_width = cx2 - cx1
    # table_width = tx2 - tx1  # 필요하면 table 폭 대비로 계산 가능

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
    1) table 아래쪽 캡션 후보 중 가장 가까운 하나를 찾는다.
    2) 아래쪽이 하나도 없으면, 위쪽 캡션 후보 중 가장 가까운 하나를 찾는다.

    * 단, x축으로 min_overlap_ratio(기본 30%) 이상 겹쳐야 후보로 인정

    Args:
        table_box (list[int]): [x1, y1, x2, y2]
        table_caption_boxes (list[list[int]]): [[x1,y1,x2,y2], ...]
        max_distance (int): 세로 방향 (table와 caption 사이) 최대 허용 거리
        min_overlap_ratio (float): x축 겹침 비율 임계값 (0 ~ 1)

    Returns:
        (best_box, best_dist, pos)
        - best_box: [x1, y1, x2, y2] or None
        - best_dist: 세로거리(float) or float('inf')
        - pos: "bottom" or "top" or ""
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
                # x축 겹침 검사
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

    # 1) 아래쪽 후보가 있으면 가장 가까운 것 선택
    if bottom_candidates:
        best_box, best_distance = bottom_candidates[0]
        pos = "bottom"
    else:
        # 2) 아래쪽이 아예 없으면, 위쪽 중 가장 가까운 것
        if top_candidates:
            best_box, best_distance = top_candidates[0]
            pos = "top"

    return best_box, best_distance, pos


def crop_region(orig_np: np.ndarray, bbox):
    """
    bounding box [x1, y1, x2, y2]에 해당하는 영역을 잘라서 return (RGB)
    orig_np는 PIL에서 np.array 변환한 (H,W,3) RGB
    """
    x1, y1, x2, y2 = bbox
    return orig_np[y1:y2, x1:x2, :]


def process_pdf(imgPath):
    """
    PDF 파일에서 figure/table과 그 caption을 분석하여,
    { caption_text: np.array(...) } 형태의 dict를 반환
    """

    # 결과를 담을 dictionary (caption -> image[numpy])
    caption_image_dict = {}

    # PDF -> Image
    pdf_file = imgPath  # 인자로 받은 PDF 경로
    pages = convert_from_path(pdf_file, 300)  # DPI=300 (필요시 조절)

    for page_idx, pil_image in enumerate(pages):
        # PIL -> numpy (RGB)
        orig_np = np.array(pil_image)

        # 2) doclayout_yolo 추론
        det_res = model.predict(
            pil_image,
            imgsz=1024,
            conf=0.3,
            device="cuda:0",
            iou=0.6
        )
        # YOLO 추론 결과를 시각화
        # det_res는 여러 이미지(batch) 가능하지만, 여기서는 1장만 처리 -> det_res[0]
        annotated_frame = det_res[0].plot(
            pil=True,  # PIL 형식으로 결과를 그릴지 여부
            line_width=3,  # 바운딩박스 선 두께
            font_size=120  # 라벨 폰트 크기
        )
        image_np = np.array(annotated_frame)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # window_name = f"Page {page_idx}"
        #
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_name, image_bgr)
        # key = cv2.waitKey(0)
        # if key == 27:
        #     break
        result = det_res[0]

        figure_boxes = []
        caption_boxes = []
        table_boxes = []
        table_caption_boxes = []

        # 3) 클래스별 분류
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

        # 4) figure - figure_caption 매핑
        for i, fig_box in enumerate(figure_boxes):
            best_box, dist, pos = find_table_caption_with_priority(fig_box, caption_boxes)

            # figure 이미지 crop (RGB)
            fig_crop = crop_region(orig_np, fig_box)

            # caption OCR
            if best_box is not None:
                caption_img = crop_region(orig_np, best_box)
                caption_text = ocr_text_from_crop(caption_img)
                print(f"매칭된 figure 캡션: {caption_text} (pos={pos})")
                # 사전에 저장 (caption -> figure image)
                caption_image_dict[caption_text] = fig_crop
            else:
                print("테이블 캡션 미발견")

        # 5) table - table_caption 매핑
        for j, tab_box in enumerate(table_boxes):
            best_box, dist, pos = find_table_caption_with_priority(tab_box, table_caption_boxes)
            tab_crop = crop_region(orig_np, tab_box)

            # caption OCR
            if best_box is not None:
                caption_img = crop_region(orig_np, best_box)
                caption_text = ocr_text_from_crop(caption_img)
                print(f"매칭된 table 캡션: {caption_text} (pos={pos})")
                caption_image_dict[caption_text] = tab_crop
            else:
                print("테이블 캡션 미발견")

    # 모든 페이지 처리 후, 키-값 형태(caption -> 이미지) 반환
    return caption_image_dict


app = Flask(__name__)


@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('mainWeb.html')


@app.route('/download/<filename>')
def download_from_memory(filename):
    """
    메모리에 저장된 이미지를 다운로드(또는 브라우저 표시)하는 라우트
    """
    if filename not in memory_store:
        abort(404, description="해당 ID에 해당하는 파일이 없습니다.")

    img_bytes = memory_store[filename]
    memfile = BytesIO(img_bytes)
    memfile.seek(0)

    # send_file 사용:
    return send_file(
        memfile,
        mimetype='image/png',
        as_attachment=True,
        download_name=f"{filename}.png"  # 다운로드될 파일 이름
    )


@app.route('/download-zip/<filename>')
def download_zip(filename: str):
    try:
        # filename이 예: 'N0890120404_IMG_...' 라면, base_name='N0890120404' 추출
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

        # ZIP 생성 완료
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


from flask import request, jsonify, url_for, abort
import os
import cv2
import tempfile


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    PDF 파일을 업로드받아 process_pdf를 통해
    {caption: np.array(...) (RGB 이미지)} 형태의 딕셔너리를 얻은 뒤,
    static 디렉토리에 png 파일로 저장하고 JSON 응답을 반환합니다.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400

        pdf_file = request.files['file']
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': '유효하지 않은 파일 형식입니다. PDF 파일을 업로드해주세요.'}), 400

        # 임시 PDF 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as temp_pdf:
            pdf_file.save(temp_pdf.name)
            temp_pdf_path = temp_pdf.name

        # PDF 처리 - 데이터 추출
        extracted_dict = process_pdf(temp_pdf_path)

        # 임시 파일 삭제
        os.remove(temp_pdf_path)

        # 저장 디렉토리
        os.makedirs(SEGMENTS_DIR, exist_ok=True)
        output_data = []

        base_filename = os.path.splitext(pdf_file.filename)[0]

        # 딕셔너리 {caption: image_np}를 순회
        # enumerate로 순서(index)도 뽑아서 파일명 구분
        for idx, (caption, img_rgb) in enumerate(extracted_dict.items()):
            # process_pdf 결과가 RGB라고 가정 → OpenCV 저장 위해 BGR 변환
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            success, encoded_img = cv2.imencode(".png", img_bgr)
            if success:
                img_bytes = encoded_img.tobytes()

                # 고유 식별자 생성 후 memory_store에 저장
                file_id = base_filename + '_IMG_' + str(uuid.uuid4())
                memory_store[file_id] = img_bytes

                #  Flask URL 생성
                download_url = url_for('download_from_memory', filename=file_id, _external=True)

                # 응답 데이터에 url, caption 기록
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
        logger.error(f"[upload_pdf] 업로드 처리 중 에러: {e}")
        return jsonify({'error': '내부 서버 오류입니다.'}), 500


# -----------------------------------------------------------------------------
# 6. Flask 앱 실행
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
