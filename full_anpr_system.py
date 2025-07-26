import cv2
import numpy as np
import os
import re
from ultralytics import YOLO
from typing import Optional, Tuple
import easyocr

# Cấu hình EasyOCR
reader = easyocr.Reader(['vi', 'en'])  # Thêm 'vi' để hỗ trợ tiếng Việt

def format_vietnam_plate(text: str) -> str:
    """Định dạng văn bản biển số Việt Nam với kiểm tra linh hoạt hơn."""
    text = re.sub(r'[^A-Z0-9.-]', '', text.upper())
    replacements = {'O': '0', 'Q': '0', 'I': '1'}
    corrected = ''.join(replacements.get(c, c) for c in text)
    # Kiểm tra linh hoạt hơn: chấp nhận 2 số + ký tự + 4 số hoặc các biến thể
    if len(corrected) >= 6 and corrected[2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        return corrected[:2] + '-' + corrected[2]  # Thêm - nếu có ký tự ở vị trí 3
    return corrected if len(corrected) >= 6 else "N/A"

def ultimate_license_plate_pipeline(plate_image: np.ndarray, output_dir: str = "test_results",
                                   debug: bool = False) -> Tuple[np.ndarray, np.ndarray, str]:
    """Xử lý ảnh biển số: cắt, tiền xử lý nhẹ với bilateral filter + Otsu, và OCR với EasyOCR."""
    if plate_image is None or plate_image.size == 0:
        print("Ảnh đầu vào không hợp lệ")
        return None, None, "N/A"

    # Resize nếu ảnh quá nhỏ
    height, width = plate_image.shape[:2]
    if height < 50 or width < 150:
        aspect_ratio = width / height
        new_width = 600
        new_height = int(new_width / aspect_ratio)
        plate_image = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    if debug:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "debug_original.jpg"), plate_image)

    # Tiền xử lý với bilateral filter
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 20, 20)  # Tăng sigma
    _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "debug_binary.jpg"), thresh)

    # Tìm contour và cắt ROI
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate_roi = plate_image
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 40
        plate_roi = plate_image[max(0, y - padding):min(y + h + padding, plate_image.shape[0]),
                    max(0, x - padding):min(x + w + padding, plate_image.shape[1])]

    if debug:
        cv2.imwrite(os.path.join(output_dir, "debug_roi.jpg"), plate_roi)

    # OCR với EasyOCR
    try:
        result = reader.readtext(plate_roi)
        print(f"Raw OCR result: {result}")  # Debug kết quả thô
        if result:
            # Kết hợp tất cả chuỗi từ EasyOCR
            tess_text = ' '.join([item[-2] for item in result])  # Ghép '60*12' và '9999'
            print(f"Combined text: {tess_text}")
        else:
            tess_text = ""
        plate_text = format_vietnam_plate(tess_text) if len(tess_text) >= 3 else "N/A"
        # Hậu xử lý
        if plate_text == "N/A" and len(tess_text) >= 3:
            parts = tess_text.split()
            if len(parts) >= 2 and '*' in parts[0]:
                corrected = parts[0].replace('*12', 'X2') + ' ' + ' '.join(parts[1:])
                plate_text = corrected  # Bỏ qua format_vietnam_plate ở bước này
            elif re.match(r'^\d{2}\d{2}\d{4}$', tess_text.replace(' ', '')):
                plate_text = tess_text[:2] + '-X' + tess_text[4:]  # Thêm -X nếu phát hiện 2 số + 2 số + 4 số
    except Exception as e:
        print(f"Lỗi OCR: {e}")
        plate_text = "N/A"

    return plate_roi, thresh, tess_text

def process_single_image(model_path: str, input_file: str, output_dir: str = "test_results",
                        debug: bool = False) -> None:
    """Xử lý ảnh với YOLO để phát hiện biển số, sau đó tiền xử lý và OCR."""
    print(f">>> BẮT ĐẦU CHẠY PIPELINE CHO ẢNH: {input_file} <<<")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[LỖI] Không thể tải mô hình YOLO: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    original_image = cv2.imread(input_file)
    if original_image is None:
        print(f"[LỖI] Không thể đọc ảnh: {input_file}")
        return

    original_image = cv2.resize(original_image, (416, 416))
    detection_results = model(original_image, conf=0.4, iou=0.5)
    plate_texts = []

    for i, result in enumerate(detection_results[0].boxes.data):
        x1, y1, x2, y2, conf, _ = map(float, result)
        plate_crop = original_image[int(y1):int(y2), int(x1):int(x2)]
        plate_roi, thresh, plate_text = ultimate_license_plate_pipeline(plate_crop, output_dir, debug)

        plate_texts.append((plate_text, conf))

        if debug:
            label = f"{plate_text} (Conf: {conf:.2f})"
            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(original_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            base_name = f"{os.path.splitext(os.path.basename(input_file))[0]}_plate_{i}"
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_ocr_{plate_text.replace('-', '_')}.jpg"), plate_roi)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.jpg"), thresh)

    output_image_path = os.path.join(output_dir, f"result_{os.path.basename(input_file)}")
    cv2.imwrite(output_image_path, original_image)

    print(f"\n>>> KẾT QUẢ NHẬN DẠNG CHO ẢNH: {os.path.basename(input_file)} <<<")
    if not plate_texts:
        print("[CẢNH BÁO] Không phát hiện được biển số nào")
    else:
        for i, (text, conf) in enumerate(plate_texts, 1):
            print(f"Biển số {i}: {text} (Độ tin cậy: {conf:.2f})")

    print(f"\nKết quả đã lưu tại: {output_dir}")

if __name__ == "__main__":
    model_path = r"D:\Uni\XLA\XuLyAnh\yolo8m\runs\yolo_bien_so_xe_detector\weights\best.pt"
    input_file = r"D:\Uni\XLA\XuLyAnh\dataset\images\0000_02187_b.jpg"
    process_single_image(model_path, input_file, debug=True)