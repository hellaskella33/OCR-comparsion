from paddleocr import PaddleOCR

# List of model configurations
models = [
    {"det_model": "en_PP-OCRv3_det_slim", "rec_model": "en_PP-OCRv3_rec_slim"},
    {"det_model": "en_PP-OCRv3_det", "rec_model": "en_PP-OCRv3_rec"},
    {"det_model": "ml_PP-OCRv3_det_slim", "rec_model": None},  # Assuming no specific rec model for multilingual slim
    {"det_model": "ml_PP-OCRv3_det", "rec_model": None},  # Assuming no specific rec model for multilingual
    {"det_model": None, "rec_model": "en_number_mobile_slim_v2.0_rec"},
    {"det_model": None, "rec_model": "en_number_mobile_v2.0_rec"},
    {"det_model": None, "rec_model": "ch_ppocr_mobile_slim_v2.0_cls"},
    {"det_model": None, "rec_model": "ch_ppocr_mobile_v2.0_cls"}
]

# Download and initialize models
for model in models:
    ocr = PaddleOCR(det_model_dir=model["det_model"] if model["det_model"] else None,
                    rec_model_dir=model["rec_model"] if model["rec_model"] else None,
                    use_gpu=False)  # Set use_gpu to True if GPU is available
    print(f"Downloaded and set up {model['det_model']} and {model['rec_model']}")
