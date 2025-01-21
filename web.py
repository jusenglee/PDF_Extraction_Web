import gc
import re
import zipfile

import easyocr
from flask import Flask, request, send_file, render_template, jsonify, url_for, send_from_directory
import os
from io import BytesIO
import layoutparser as lp
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract

app = Flask(__name__)

LOCAL_CONFIG_PATH = "./config.yml"
LOCAL_PTH_PATH = "./model_final.pth"
model = lp.Detectron2LayoutModel(
    config_path=LOCAL_CONFIG_PATH,
    label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
    extra_config=["MODEL.WEIGHTS", LOCAL_PTH_PATH, "MODEL.DEVICE", "cuda"]
)
reader = easyocr.Reader(['en', 'ko'])



def process_pdf_with_layoutparser(pdf_path, dpi=300):
    extracted_images = []
    images = convert_from_path(pdf_path, dpi=dpi)
    for page_number, image in enumerate(images):
        print(f"Processing page {page_number + 1}...")
        layout = model.detect(image)
        filtered_blocks = [l for l in layout if l.type in ["table", "figure"] and l.score > 0.80]

        image_np = np.array(image)
        rgb_img = image_np

        mask_aggregate = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)  # 페이지 단위로 초기화

        for l_obj in filtered_blocks:
            x1, y1, x2, y2 = map(int, [l_obj.block.x_1, l_obj.block.y_1, l_obj.block.x_2, l_obj.block.y_2])
            segment_rgb = rgb_img[y1:y2, x1:x2]

            mask_binary = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
            mask_binary[y1:y2, x1:x2] = 1

            overlap = cv2.bitwise_and(mask_binary, mask_aggregate)
            overlap_ratio = np.sum(overlap) / np.sum(mask_binary) if np.sum(mask_binary) > 0 else 0

            if overlap_ratio > 0.1:
                continue

            mask_aggregate = cv2.bitwise_or(mask_aggregate, mask_binary)
            mask_cropped = mask_binary[y1:y2, x1:x2]

            kernel = np.ones((3, 3), np.uint8)
            mask_cropped = cv2.morphologyEx(mask_cropped, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_cropped = cv2.morphologyEx(mask_cropped, cv2.MORPH_DILATE, kernel, iterations=1)

            segment_rgba = cv2.cvtColor(segment_rgb, cv2.COLOR_BGR2BGRA)
            segment_rgba[:, :, 3] = (mask_cropped * 255).astype(np.uint8)
            extracted_images.append(segment_rgba)

        # 메모리 해제
        del image_np, rgb_img
        gc.collect()

    return extracted_images



def extract_text_from_image(segment_rgb):
    gray_image = cv2.cvtColor(segment_rgb, cv2.COLOR_RGB2GRAY)
    text = reader.readtext(gray_image, detail=0, paragraph=True)
    print(text)
    return ' '.join(text)


@app.route('/')
def index():
    return render_template('./mainWeb.html')


@app.route('/download/<filename>')
def download_file(filename):
    """Endpoint to download an individual file."""
    segments_dir = "./static/segments"
    return send_from_directory(segments_dir, filename, as_attachment=True)

@app.route('/download-zip/<filename>')
def download_zip(filename):
    """Endpoint to download all images as a ZIP file."""
    segments_dir = "./static/segments"
    base_name = filename.split('_')[0]
    print("download-zip : target - ", base_name)

    zip_buffer = BytesIO()

    # Create a ZIP file in memory
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # 파일명 패턴 정의: base_name_index 형식
        pattern = re.compile(rf"^{base_name}_\d+$")

        matching_files = [
            file for file in os.listdir(segments_dir)
            if pattern.match(os.path.splitext(file)[0])  # 파일 확장자 제외하고 매칭
        ]

        if not matching_files:
            return "No matching files found for the given base name.", 404

        for file in matching_files:
            full_path = os.path.join(segments_dir, file)
            if os.path.isfile(full_path):  # Ensure it's a file
                zip_file.write(full_path, arcname=file)

    zip_buffer.seek(0)
    zip_filename = f"{base_name}_images.zip"

    # Send the ZIP file to the client
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )




@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    pdf_file = request.files['file']
    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

    temp_pdf_path = "./temp.pdf"
    pdf_file.save(temp_pdf_path)
    uploaded_filename = pdf_file.filename.split('.')[0]
    extracted_images = process_pdf_with_layoutparser(temp_pdf_path)
    os.remove(temp_pdf_path)

    output_urls = []
    segments_dir = "./static/segments"
    os.makedirs(segments_dir, exist_ok=True)

    for idx, img in enumerate(extracted_images):
        filename = f"{os.path.splitext(pdf_file.filename)[0]}_{idx}.png"
        out_path = os.path.join(segments_dir, filename)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

        url = url_for('static', filename=f"segments/{filename}")
        output_urls.append(url)

    return jsonify({'images': output_urls, 'uploaded_filename': uploaded_filename}), 200

if __name__ == '__main__':
    app.run(debug=True)
