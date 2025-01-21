import os
import re
import gc
import hashlib
import zipfile
import logging
import tempfile
from io import BytesIO
from typing import List, Dict, Any

from flask import (
    Flask,
    request,
    send_file,
    render_template,
    jsonify,
    url_for,
    send_from_directory,
    abort
)
import layoutparser as lp
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract

# -----------------------------------------------------------------------------
# 1. 환경 변수 로드 (dotenv 사용을 권장하지만, 여기서는 가정만 해 둠)
# -----------------------------------------------------------------------------
CACHE_DIR = os.getenv('CACHE_DIR', "./cache")
CONFIG_PATH = os.getenv('CONFIG_PATH', "./config.yml")
MODEL_PATH = os.getenv('MODEL_PATH', "./model_final.pth")
SEGMENTS_DIR = os.getenv('SEGMENTS_DIR', "./static/segments")
TEMP_DIR = os.getenv('TEMP_DIR', tempfile.gettempdir())

# -----------------------------------------------------------------------------
# 2. Logging 설정
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 3. OCR 에이전트 및 LayoutParser 모델 로드
# -----------------------------------------------------------------------------
ocr_agent = lp.TesseractAgent(languages='kor+eng')

try:
    # detectron2 모델 로드
    device = "cuda"  # 또는 환경 변수, torch.cuda.is_available() 체크 등으로 동적 설정 가능
    model = lp.Detectron2LayoutModel(
        config_path=CONFIG_PATH,
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
        extra_config=[
            "MODEL.WEIGHTS", MODEL_PATH,
            "MODEL.DEVICE", "cuda",
             "MODEL.ROI_HEADS.SCORE_THRESH_TEST", "0.7",  # 점수 기준
             "MODEL.ROI_HEADS.NMS_THRESH_TEST", "0.2"    # NMS 임계값
        ]
    )
    logger.info("LayoutParser 모델 로드 완료.")
except Exception as e:
    logger.error(f"LayoutParser 모델 로드 실패: {e}")
    raise

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract')


# -----------------------------------------------------------------------------
# 4. PDF 처리 및 OCR 관련 함수
# -----------------------------------------------------------------------------
def process_pdf(pdf_path: str, dpi: int = 250) -> List[Dict[str, Any]]:
    """
    PDF 파일을 처리하여 테이블, 그림(figure) 영역을 추출하고,
    주변 캡션을 OCR로 추출한 뒤 결과 리스트를 반환합니다.

    Args:
        pdf_path (str): 처리할 PDF 파일의 경로
        dpi (int, optional): PDF -> Image 변환 시 해상도. 기본 300.

    Returns:
        List[Dict[str, Any]]: [
            {
                'image': np.ndarray,  # BGR + Alpha(RGBA)
                'caption': str        # OCR로 추출된 캡션
            },
            ...
        ]
    """
    extracted_images = []
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        for page_idx, pil_image in enumerate(images):
            logger.info(f"[process_pdf] 페이지 {page_idx + 1}/{len(images)} 처리 중...")

            # Layout 추론
            layout = model.detect(pil_image)
            filtered_blocks = [
                l for l in layout
                if l.type in ("table", "figure")
            ]
            text_blocks = [
                l for l in layout
                if l.type == "text"
            ]

            # PIL -> OpenCV 배열
            image_np = np.array(pil_image)
            bgr_img = image_np

            # Segmentation 마스크(중복 방지용)
            mask_aggregate = np.zeros((bgr_img.shape[0], bgr_img.shape[1]), dtype=np.uint8)

            for block in filtered_blocks:
                x1, y1, x2, y2 = map(int, [
                    block.block.x_1,
                    block.block.y_1,
                    block.block.x_2,
                    block.block.y_2
                ])
                segment_bgr = bgr_img[y1:y2, x1:x2]

                # 중복 확인용 마스크 생성
                mask_binary = np.zeros_like(mask_aggregate)
                mask_binary[y1:y2, x1:x2] = 1

                overlap = cv2.bitwise_and(mask_binary, mask_aggregate)
                overlap_ratio = 0.0
                if np.sum(mask_binary) > 0:
                    overlap_ratio = np.sum(overlap) / np.sum(mask_binary)

                # 중복이 50% 이상이면 스킵
                if overlap_ratio > 0.6:
                    logger.debug("[process_pdf] 중복된 영역 발견, 건너뜀.")
                    continue

                # 캡션 추출
                caption = extract_caption_bottom_priority(block.block, text_blocks, bgr_img)

                # 마스크 병합
                mask_aggregate = cv2.bitwise_or(mask_aggregate, mask_binary)

                # 세그먼트 영역 마스크
                mask_cropped = mask_binary[y1:y2, x1:x2]

                # 모폴로지 연산으로 노이즈 제거
                kernel = np.ones((3, 3), np.uint8)
                mask_cropped = cv2.morphologyEx(mask_cropped, cv2.MORPH_OPEN, kernel, iterations=1)
                mask_cropped = cv2.morphologyEx(mask_cropped, cv2.MORPH_DILATE, kernel, iterations=1)

                # RGBA 변환
                segment_rgba = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2BGRA)
                segment_rgba[:, :, 3] = (mask_cropped * 255).astype(np.uint8)

                extracted_images.append({
                    'image': segment_rgba,
                    'caption': caption
                })

            # 메모리 해제
            del image_np, bgr_img
            gc.collect()

    except Exception as e:
        logger.error(f"[process_pdf] PDF 처리 중 에러: {e}")
        raise

    return extracted_images


def extract_caption_bottom_priority(
    figure_bbox: Any,
    text_blocks: List[lp.Layout],
    bgr_img: np.ndarray,
    max_distance: int = 100
) -> str:
    """
    1) 하단에서 먼저 캡션을 찾고
    2) 찾지 못하면 상단에서 찾는 로직.

    Returns:
        str: 최종 캡션(문자열). 없으면 빈 문자열
    """
    figure_bottom = figure_bbox.y_2
    figure_top = figure_bbox.y_1

    # "Figure 1.", "Fig. 2.", "Table 3." 등
    caption_pattern = re.compile(r'^(Fig\.|Figure|Tab\.|Table)\s*\d+', re.IGNORECASE)

    # 특정 거리 이내에 있는 text_blocks만 필터링 (성능 개선 목적)
    candidate_blocks = filter_text_blocks_by_distance(
        figure_bbox,
        text_blocks,
        max_distance
    )

    best_caption = ""

    # 먼저 하단에서 찾는다
    for block in candidate_blocks:
        distance_bottom = block.block.y_1 - figure_bottom
        # block이 figure 바로 아래(거리가 0~max_distance)인지 확인
        if 0 <= distance_bottom <= max_distance:
            text = ocr_extract_text(block, bgr_img)
            if text and caption_pattern.match(text):
                best_caption = text.strip()
                break  # 하단에서 찾았으면 즉시 반환

    # 하단에서 캡션을 찾지 못했을 경우 상단을 확인
    if not best_caption:
        for block in candidate_blocks:
            distance_top = figure_top - block.block.y_2
            if 0 <= distance_top <= max_distance:
                text = ocr_extract_text(block, bgr_img)
                if text and caption_pattern.match(text):
                    best_caption = text.strip()
                    break

    return best_caption


def filter_text_blocks_by_distance(
        figure_bbox: Any,
        text_blocks: List[lp.Layout],
        max_distance: int
) -> List[lp.Layout]:
    """
    figure_bbox로부터 세로(또는 중심점) 거리가 max_distance 이하인
    텍스트 블록만 리턴합니다.
    """
    figure_bottom = figure_bbox.y_2
    figure_top = figure_bbox.y_1

    candidate_blocks = []
    for block in text_blocks:
        if block.type != "text":
            continue

        # 블록 위/아래 거리 계산
        distance_bottom = block.block.y_1 - figure_bottom
        distance_top = figure_top - block.block.y_2

        # 세로 거리 기준으로 캡션 후보로 삼을지 결정
        if 0 <= distance_bottom <= max_distance or 0 <= distance_top <= max_distance:
            candidate_blocks.append(block)

    return candidate_blocks


def ocr_extract_text(block: lp.Layout, bgr_img: np.ndarray) -> str:
    """
    주어진 LayoutParser 블록 영역을 OpenCV로 크롭 후 OCR로 텍스트를 추출.
    (이미 기존 코드에 있으나, 여기서는 예시로 재작성)
    """
    try:
        segment_image = (
            block
            .pad(left=5, right=5, top=5, bottom=5)
            .crop_image(bgr_img)
        )
        gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.medianBlur(thresh, 3)

        text = ocr_agent.detect(denoised)
        extracted_text = text.strip()

        logger.debug(f"[ocr_extract_text] OCR 추출 텍스트: {extracted_text}")
        return extracted_text

    except Exception as e:
        logger.error(f"[ocr_extract_text] OCR 추출 중 에러: {e}")
        return ""


# -----------------------------------------------------------------------------
# 5. Flask 앱 초기화 및 라우팅
# -----------------------------------------------------------------------------
app = Flask(__name__)


@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('mainWeb.html')


@app.route('/download/<filename>')
def download_file(filename: str):
    """
    개별 파일 다운로드 엔드포인트.
    """
    try:
        # 파일명 검증
        if not re.match(r'^[\w,\s-]+\.[A-Za-z]{3,4}$', filename):
            abort(400, description="잘못된 파일명입니다.")

        return send_from_directory(SEGMENTS_DIR, filename, as_attachment=True)

    except FileNotFoundError:
        abort(404, description="파일을 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"[download_file] 파일 다운로드 중 에러: {e}")
        abort(500, description="내부 서버 오류입니다.")


@app.route('/download-zip/<filename>')
def download_zip(filename: str):
    """
    모든 이미지를 ZIP 파일로 다운로드하는 엔드포인트.
    filename은 실제 PDF 파일 이름 등과 연결되어
    <base_name>_n.png 형태로 저장된 이미지를 ZIP으로 묶습니다.
    """
    try:
        base_name = filename.split('_')[0]
        logger.info(f"[download_zip] ZIP 다운로드 요청: base_name={base_name}")

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            pattern = re.compile(rf"^{re.escape(base_name)}_\d+\.[A-Za-z]{{3,4}}$")
            for file in os.listdir(SEGMENTS_DIR):
                if pattern.match(file):
                    full_path = os.path.join(SEGMENTS_DIR, file)
                    if os.path.isfile(full_path):
                        zip_file.write(full_path, arcname=file)

        # 실제로 ZIP에 담긴 파일이 없으면 404
        if zip_buffer.getbuffer().nbytes == 0:
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


@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    PDF 파일을 업로드받은 뒤, 테이블/그림 세그먼트를 추출하고
    static 디렉토리에 저장. 그 정보를 JSON으로 반환합니다.
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

        # PDF 처리
        extracted_images = process_pdf(temp_pdf_path)

        # 임시 파일 삭제
        os.remove(temp_pdf_path)

        # 세그먼트 이미지 저장
        os.makedirs(SEGMENTS_DIR, exist_ok=True)
        output_data = []

        base_filename = os.path.splitext(pdf_file.filename)[0]

        for idx, img_data in enumerate(extracted_images):
            img = img_data['image']
            caption = img_data['caption']

            filename = f"{base_filename}_{idx}.png"
            out_path = os.path.join(SEGMENTS_DIR, filename)
            cv2.imwrite(out_path, img)

            file_url = url_for('download_file', filename=filename)
            output_data.append({
                'url': file_url,
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
