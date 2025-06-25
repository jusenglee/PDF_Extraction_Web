# PDF Extraction Web

이 프로젝트는 PDF 파일에서 테이블(Table)과 그림(Figure)을 감지하여 PNG 이미지로 추출하고, 각 이미지 주변의 캡션을 OCR로 인식하는 Flask 기반의 웹 애플리케이션입니다. 업로드한 PDF를 분석하여 세그먼트 이미지를 다운로드하거나 ZIP으로 묶어 받을 수 있습니다.

## 특징
- [LayoutParser](https://github.com/Layout-Parser/layout-parser)와 Detectron2 모델을 사용하여 테이블/그림 영역 감지
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)을 활용한 캡션 인식
- 간단한 웹 UI를 통한 PDF 업로드 및 결과 이미지 다운로드

## 설치 방법
1. Python 3.10 환경을 준비합니다.
2. 의존성 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
3. 모델 가중치를 다운로드하여 `model_final.pth` 이름으로 저장합니다. 기본 설정은 [HuggingFace](https://huggingface.co/tensorlake/faster_rcnn-publaynet/tree/main) 또는 config 파일의 링크를 사용합니다.

## 실행 방법
1. 웹 서버 실행
   ```bash
   python web.py
   ```
2. 브라우저에서 `http://localhost:5000` 주소로 접속하여 PDF 파일을 업로드합니다.
3. 추출된 이미지와 OCR로 인식한 캡션을 확인하고, 개별 PNG 또는 ZIP 파일로 다운로드합니다.

## 환경 변수
- `CACHE_DIR`, `CONFIG_PATH`, `MODEL_PATH`, `SEGMENTS_DIR`, `TEMP_DIR`, `TESSERACT_CMD` 등을 필요에 따라 조정할 수 있습니다. 기본값은 `web.py`에 정의되어 있습니다.

## 폴더 구조
- `detectron2/` : Detectron2 라이브러리 소스
- `templates/` : Flask 템플릿(`mainWeb.html`)
- `static/` : 자바스크립트(`script.js`)와 스타일(`styles.css`), 추출된 이미지 저장 위치
- `config.yml` : 모델 설정 파일
- `web.py` : Flask 애플리케이션 메인 파일

## 스크린샷
![image](https://github.com/user-attachments/assets/bfe5e155-fe6c-4889-910b-77ad61276444)

