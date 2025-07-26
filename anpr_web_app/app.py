import os
import re
import uuid
import cv2
from flask import Flask, render_template, request, jsonify, session
from anpr_core import ANPRSystem

# --- Cấu hình ---
YOLO_MODEL_PATH = r"E:\XLA\XuLyAnh\runs\yolo_bien_so_xe_detector\weights\best.pt" 
RESULT_FOLDER = 'results' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'super-secret-key-for-session-and-temp-storage'

# --- Tải mô hình ANPR (CHỈ MỘT LẦN KHI START SERVER) ---
try:
    anpr_system = ANPRSystem(yolo_model_path=YOLO_MODEL_PATH)
except Exception as e:
    print(f"Không thể khởi tạo ANPR System. Ứng dụng sẽ thoát. Lỗi: {e}")
    exit()


# --- Nơi lưu trữ tạm thời kết quả xử lý (trong bộ nhớ server) ---
# Dùng dictionary để lưu, key là session_id, value là kết quả xử lý
TEMP_RESULTS_STORAGE = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Chỉ hiển thị trang chính."""
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    """API để xử lý ảnh và trả về kết quả dạng JSON."""
    if 'image' not in request.files:
        return jsonify({'error': 'Không có file ảnh nào được gửi lên.'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'File không hợp lệ hoặc không được cho phép.'}), 400

    try:
        image_bytes = file.read()
        results_data = anpr_system.process_image_in_memory(image_bytes)

        if "error" in results_data:
            return jsonify({'error': results_data['error']}), 500

        # Tạo một session ID duy nhất cho lần xử lý này
        session_id = str(uuid.uuid4())
        
        # Lưu kết quả (dạng NumPy array) vào bộ nhớ tạm
        TEMP_RESULTS_STORAGE[session_id] = results_data.copy()

        # Chuẩn bị dữ liệu để gửi về client (chuyển ảnh sang Base64)
        response_data = {
            'session_id': session_id,
            'result_image_base64': ANPRSystem.encode_image_to_base64(results_data['result_image_np']),
            'plates': []
        }

        for plate in results_data['plates']:
            response_data['plates'].append({
                'text': plate['text'],
                'confidence': plate['confidence'],
                'cropped_plate_base64': ANPRSystem.encode_image_to_base64(plate['cropped_plate_np']),
                'binary_plate_base64': ANPRSystem.encode_image_to_base64(plate['binary_plate_np'])
            })
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Đã xảy ra lỗi không xác định: {e}")
        return jsonify({'error': f'Xảy ra lỗi trong quá trình xử lý ảnh: {e}'}), 500

@app.route('/save-results', methods=['POST'])
def save_results():
    """API để lưu kết quả từ bộ nhớ tạm ra file."""
    data = request.get_json()
    session_id = data.get('session_id')
    
    if not session_id or session_id not in TEMP_RESULTS_STORAGE:
        return jsonify({'status': 'error', 'message': 'Không tìm thấy dữ liệu để lưu hoặc phiên đã hết hạn.'}), 404

    try:
        results_to_save = TEMP_RESULTS_STORAGE[session_id]
        
        # Tạo thư mục con trong 'results' với tên là session_id
        session_output_dir = os.path.join(app.config['RESULT_FOLDER'], session_id)
        os.makedirs(session_output_dir, exist_ok=True)

        # Lưu ảnh kết quả chính
        result_image_filename = "result_image_with_box.jpg"
        result_image_path = os.path.join(session_output_dir, result_image_filename)
        cv2.imwrite(result_image_path, results_to_save['result_image_np'])

        # Lưu các ảnh biển số đã xử lý
        for i, plate in enumerate(results_to_save['plates']):
            text_for_filename = re.sub(r'[^A-Z0-9]', '_', plate['text'])
            
            cropped_filename = f"plate_{i}_{text_for_filename}_cropped.jpg"
            cropped_path = os.path.join(session_output_dir, cropped_filename)
            cv2.imwrite(cropped_path, plate['cropped_plate_np'])
            
            if plate.get('binary_plate_np') is not None:
                binary_filename = f"plate_{i}_{text_for_filename}_binary.jpg"
                binary_path = os.path.join(session_output_dir, binary_filename)
                cv2.imwrite(binary_path, plate['binary_plate_np'])
        
        # Xóa dữ liệu khỏi bộ nhớ tạm để giải phóng tài nguyên
        del TEMP_RESULTS_STORAGE[session_id]

        return jsonify({'status': 'success', 'message': f'Kết quả đã được lưu thành công vào thư mục: results/{session_id}'})

    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")
        return jsonify({'status': 'error', 'message': f'Đã xảy ra lỗi trong quá trình lưu file: {e}'}), 500

if __name__ == '__main__':
    # Tạo thư mục results nếu chưa có
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    app.run(debug=True, host='0.0.0.0')