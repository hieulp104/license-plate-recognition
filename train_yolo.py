import os
import random
import shutil
import yaml
from ultralytics import YOLO

# ==============================================================================
# PHẦN 1: CẤU HÌNH (BẠN CHỈ CẦN THAY ĐỔI CÁC THÔNG SỐ Ở ĐÂY)
# ==============================================================================

# 1. Đường dẫn đến thư mục dataset gốc của bạn
#    Thư mục này phải chứa 2 thư mục con là 'images' và 'labels'
SOURCE_DATASET_DIR = "dataset" 

# 2. Đường dẫn đến thư mục mới để chứa dữ liệu đã được chia (train/val)
#    Kịch bản sẽ tự tạo thư mục này nếu nó chưa tồn tại.
SPLIT_DATASET_DIR = "dataset_split"

# 3. Tỷ lệ dữ liệu dành cho tập kiểm tra (validation). 
#    0.2 nghĩa là 80% cho training, 20% cho validation.
VALIDATION_SPLIT = 0.2

# 4. Tên các lớp đối tượng của bạn.
#    Thứ tự phải khớp với class_id trong các file nhãn .txt.
#    Nếu bạn chỉ có một lớp là "biển số xe" (class_id = 0), hãy để như sau:
CLASS_NAMES = ["lisence_plate"]

# 5. Các tham số cho quá trình huấn luyện YOLO
EPOCHS = 50          # Số chu kỳ huấn luyện. Bắt đầu với 2, có thể tăng nếu cần.
IMAGE_SIZE = 640     # Kích thước ảnh đầu vào cho mô hình. 640 là giá trị phổ biến.
BATCH_SIZE = 16      # -1 để YOLO tự động điều chỉnh batch size cho phù hợp với VRAM của GPU.
YOLO_MODEL = 'yolov8x.pt' # Model để bắt đầu. 'n' (nano) là nhỏ nhất, 's' (small), 'm' (medium).

# ==============================================================================
# PHẦN 2: CÁC HÀM TIỆN ÍCH (KHÔNG CẦN THAY ĐỔI)
# ==============================================================================

def split_data(source_dir, output_dir, val_split):
    """
    Chia dữ liệu từ thư mục nguồn thành các tập train và validation.
    """
    print(">>> BƯỚC 1: BẮT ĐẦU QUÁ TRÌNH CHIA DỮ LIỆU <<<")
    
    source_images_dir = os.path.join(source_dir, "images")
    source_labels_dir = os.path.join(source_dir, "labels")

    if not os.path.isdir(source_images_dir) or not os.path.isdir(source_labels_dir):
        print(f"[LỖI] Không tìm thấy thư mục 'images' hoặc 'labels' bên trong '{source_dir}'.")
        print("Vui lòng kiểm tra lại cấu trúc thư mục của bạn.")
        return False

    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")

    # Tạo các thư mục đích
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Lấy danh sách tất cả các file ảnh và xáo trộn chúng để đảm bảo tính ngẫu nhiên
    image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)

    # Tính toán điểm chia dữ liệu
    split_index = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Hàm trợ giúp để sao chép các cặp file (ảnh và nhãn)
    def copy_files(files, dest_img_dir, dest_lbl_dir):
        count = 0
        for img_file in files:
            base_name = os.path.splitext(img_file)[0]
            lbl_file = f"{base_name}.txt"
            
            source_img_path = os.path.join(source_images_dir, img_file)
            source_lbl_path = os.path.join(source_labels_dir, lbl_file)

            if os.path.exists(source_lbl_path):
                shutil.copy(source_img_path, dest_img_dir)
                shutil.copy(source_lbl_path, dest_lbl_dir)
                count += 1
        return count

    print(f"[INFO] Đang sao chép {len(train_files)} file vào tập train...")
    num_train = copy_files(train_files, train_images_dir, train_labels_dir)
    
    print(f"[INFO] Đang sao chép {len(val_files)} file vào tập validation...")
    num_val = copy_files(val_files, val_images_dir, val_labels_dir)
    
    print("\n[THÀNH CÔNG] Chia dữ liệu hoàn tất!")
    print(f"  - Tổng số cặp ảnh/nhãn hợp lệ: {num_train + num_val}")
    print(f"  - Tập Train: {num_train} ảnh")
    print(f"  - Tập Validation: {num_val} ảnh")
    return True

def create_yaml_file(output_dir, class_names):
    """
    Tạo file cấu hình .yaml cho YOLO để nó biết dữ liệu nằm ở đâu.
    """
    print("\n>>> BƯỚC 2: TẠO FILE CẤU HÌNH DATA.YAML <<<")
    
    # Lấy đường dẫn tuyệt đối và thay thế dấu gạch chéo ngược bằng dấu gạch chéo tới
    abs_output_dir = os.path.abspath(output_dir).replace('\\', '/')

    data_config = {
        'path': abs_output_dir,  # Thư mục gốc của dữ liệu đã chia
        'train': 'train/images', # Đường dẫn tương đối từ 'path'
        'val': 'val/images',     # Đường dẫn tương đối từ 'path'
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
    print(f"[THÀNH CÔNG] File data.yaml đã được tạo tại: {yaml_path}")
    return yaml_path


def evaluate_model(model, yaml_path, output_dir):
    """
    Evaluate the trained model and save metrics.
    """
    print("\n>>> BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH <<<")
    results = model.val(data=yaml_path)

    # Extract metrics
    metrics = {
        'mAP@0.5': results.box.map50,
        'mAP@0.5:0.95': results.box.map,
        'Precision': results.box.p[0] if results.box.p else 0,
        'Recall': results.box.r[0] if results.box.r else 0,
        'F1-Score': results.box.f1[0] if results.box.f1 else 0
    }

    # Save metrics to file
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("=== Model Evaluation Results ===\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"[THÀNH CÔNG] Evaluation results saved to: {results_path}")

    # Ultralytics saves plots (e.g., confusion matrix) in the 'runs/detect' folder
    print("Confusion matrix and other plots saved in 'runs/detect/yolo_bien_so_xe_detector'")

    return metrics

# ==============================================================================
# PHẦN 3: THỰC THI CHƯƠNG TRÌNH
# ==============================================================================

if __name__ == "__main__":
    # Bước 1: Chia dữ liệu
    if not split_data(SOURCE_DATASET_DIR, SPLIT_DATASET_DIR, VALIDATION_SPLIT):
        # Dừng lại nếu không chia được dữ liệu
        exit()

    # Bước 2: Tạo file YAML
    yaml_file_path = create_yaml_file(SPLIT_DATASET_DIR, CLASS_NAMES)

    # Bước 3: Huấn luyện YOLOv8
    print("\n>>> BƯỚC 3: BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN YOLOv8 <<<")
    print("="*2)

    # Tải một mô hình YOLOv8 đã được huấn luyện trước (pre-trained)
    model = YOLO(YOLO_MODEL)

    # Bắt đầu huấn luyện (fine-tuning) trên bộ dữ liệu của bạn
    results = model.train(
        data=yaml_file_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name='yolo_bien_so_xe_detector' # Tên của thư mục kết quả huấn luyện
    )

    metrics = evaluate_model(model, yaml_file_path, SPLIT_DATASET_DIR)

    print("\n" + "="*2)
    print(">>> HUẤN LUYỆN HOÀN TẤT! <<<")
    print("Kết quả được lưu trong thư mục 'runs/detect/yolo_bien_so_xe_detector'")
    print("Mô hình tốt nhất để sử dụng là: 'runs/detect/yolo_bien_so_xe_detector/weights/best.pt'")
    print("="*2)