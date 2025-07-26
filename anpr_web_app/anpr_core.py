import cv2
import numpy as np
import os
import re
from ultralytics import YOLO
from typing import Optional, Tuple
import easyocr
import base64

class ANPRSystem:
    def __init__(self, yolo_model_path: str):
        """
        Khởi tạo hệ thống ANPR, tải các mô hình cần thiết một lần.
        """
        print("Đang tải mô hình YOLOv8...")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print("Tải mô hình YOLOv8 thành công.")
        except Exception as e:
            print(f"[LỖI NGHIÊM TRỌNG] Không thể tải mô hình YOLO: {e}")
            raise e

        print("Đang tải mô hình EasyOCR...")
        try:
            self.reader = easyocr.Reader(['vi', 'en'])
            print("Tải mô hình EasyOCR thành công.")
        except Exception as e:
            print(f"[LỖI NGHIÊM TRỌNG] Không thể tải mô hình EasyOCR: {e}")
            raise e
            
        print("Hệ thống ANPR đã sẵn sàng.")

    def _format_vietnam_plate(self, text: str) -> str:
        """Định dạng văn bản biển số Việt Nam: tất cả là số trừ vị trí thứ 3 là chữ."""
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(text) < 4:
            return "N/A"

        num_to_char = {'8': 'B', '4': 'A', '0': 'O', '1': 'I', '5': 'S', '6': 'G', '7': 'H'}
        char_to_num = {'B': '8', 'O': '0', 'I': '1', 'S': '5', 'A': '4', 'X': '8', 'J': '3', 'G': '6'}

        if len(text) >= 7:
            # Tiền tố: 2 số đầu
            prefix = text[:2]
            # Ký tự thứ 3 (mã quận/huyện)
            area_code = text[2]
            # Hậu tố: các số còn lại
            suffix = text[3:]

            # Chuẩn hóa tiền tố
            new_prefix = ''.join([char_to_num.get(c, c) for c in prefix if c.isalnum()])
            # Chuẩn hóa hậu tố
            new_suffix = ''.join([char_to_num.get(c, c) for c in suffix if c.isalnum()])
            # Chuẩn hóa mã quận
            if area_code.isdigit():
                area_code = num_to_char.get(area_code, area_code)

            # Chỉ ghép lại nếu hợp lệ
            if new_prefix.isdigit() and area_code.isalpha() and new_suffix.isdigit():
                if len(new_suffix) == 4:
                    return f"{new_prefix}{area_code}-{new_suffix[:1]}.{new_suffix[1:]}"
                elif len(new_suffix) == 5:
                    return f"{new_prefix}{area_code}-{new_suffix[:2]}.{new_suffix[2:]}"
                else:
                    return f"{new_prefix}{area_code}-{new_suffix}"

        # Nếu không khớp định dạng trên, trả về dạng đã làm sạch
        return text if text else "N/A"

    def _ultimate_license_plate_pipeline(self, image: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
        """Xử lý ảnh biển số đã cắt: tiền xử lý, và OCR."""
        if image is None or image.size == 0:
            return "N/A", None, None

        height, width = image.shape[:2]
        if height < 32 or width < 80:
            scale = max(2.0, 32 / height, 80 / width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)
        
        plate_roi = image.copy()
        
        # Tiền xử lý với CLAHE và bilateral filter
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        bfilter = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Nhị phân hóa
        thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
        
        plate_text = "N/A"
        try:
            result = self.reader.readtext(bfilter, detail=1, paragraph=False)
            if result:
                # Sắp xếp các box theo thứ tự từ trái sang phải, trên xuống dưới
                result.sort(key=lambda x: (x[0][0][1], x[0][0][0])) 
                raw_text = ''.join([item[1] for item in result])
                cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                # Kiểm tra số chữ cái
                letter_count = sum(1 for c in cleaned_text if c.isalpha())
                if letter_count >= 5:
                    return "N/A", None, None
                plate_text = self._format_vietnam_plate(raw_text)
        except Exception as e:
            print(f"Lỗi OCR: {e}")

        return plate_text, plate_roi, thresh

    def process_image_in_memory(self, image_bytes: bytes) -> dict:
        """
        Phát hiện và nhận dạng biển số từ dữ liệu byte của ảnh.
        Trả về một dictionary chứa dữ liệu ảnh (NumPy array) và văn bản.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if original_image is None:
            return {"error": "Không thể đọc ảnh."}
        
        detection_results = self.yolo_model(original_image, conf=0.4, iou=0.5)
        
        detected_plates = []
        image_with_boxes = original_image.copy()

        for i, result in enumerate(detection_results[0].boxes.data):
            box_data = result.tolist()
            x1, y1, x2, y2 = map(int, box_data[:4])
            conf = float(box_data[4])

            h, w, _ = original_image.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            
            plate_crop = original_image[y1:y2, x1:x2]
            plate_text, processed_plate_img, binary_plate_img = self._ultimate_license_plate_pipeline(plate_crop)
            
            # Chỉ thêm vào detected_plates nếu vùng được coi là biển số hợp lệ
            if processed_plate_img is not None:
                detected_plates.append({
                    "text": plate_text,
                    "confidence": conf,
                    "cropped_plate_np": processed_plate_img,
                    "binary_plate_np": binary_plate_img,
                })
                
                label = f"{plate_text} ({conf:.2f})"
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image_with_boxes, (x1, y1 - text_height - 15), (x1 + text_width, y1 - 10), (0, 255, 0), -1)
                cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return {
            "result_image_np": image_with_boxes,
            "plates": detected_plates
        }

    @staticmethod
    def encode_image_to_base64(image_np: np.ndarray, format: str = ".jpg") -> str:
        """Chuyển đổi ảnh NumPy array sang chuỗi Base64 Data URL."""
        if image_np is None:
            return None
        _, buffer = cv2.imencode(format, image_np)
        encoded_string = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/{format[1:]};base64,{encoded_string}"