import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from pdf2image import convert_from_path
import time

# Load the pre-trained YOLOv10 model
filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(filepath)
# Convert PDF pages to images
pdf_file = './ATN0050547548.pdf'
images = convert_from_path(pdf_file, 400)  # DPI=300
start = time.time()

for page_idx, pil_image in enumerate(images):
    # doclayout_yolo에 PIL 이미지를 바로 전달
    det_res = model.predict(
        pil_image,
        imgsz=2048,   # Prediction image size
        conf=0.2,     # Confidence threshold
        device="cuda:0"  # GPU 사용 ('cpu' 사용 가능)
    )
    print("det_res   :   ", det_res[0])
    # YOLO 추론 결과를 시각화
    # det_res는 여러 이미지(batch) 가능하지만, 여기서는 1장만 처리 -> det_res[0]
    annotated_frame = det_res[0].plot(
        pil=True,      # PIL 형식으로 결과를 그릴지 여부
        line_width=3,  # 바운딩박스 선 두께
        font_size=120   # 라벨 폰트 크기
    )
    print("annotated_frame   :   ", annotated_frame)
    image_np = np.array(annotated_frame)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    window_name = f"Page {page_idx}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image_bgr)

    key = cv2.waitKey(0)
    if key == 27:
        break

# 모든 창 닫기
cv2.destroyAllWindows()