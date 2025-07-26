import os

# --- CẤU HÌNH ---
# Thay đổi đường dẫn này cho đúng với thư mục dataset GỐC của bạn
DATASET_DIR = r"D:\Uni\XLA\dataset"
# -----------------

images_dir = os.path.join(DATASET_DIR, "images")
labels_dir = os.path.join(DATASET_DIR, "labels")

print(">>> BẮT ĐẦU KIỂM TRA BỘ DỮ LIỆU <<<")
print(f"Thư mục ảnh: {os.path.abspath(images_dir)}")
print(f"Thư mục nhãn: {os.path.abspath(labels_dir)}")
print("-" * 50)

if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
    print("[LỖI NGHIÊM TRỌNG] Không tìm thấy thư mục 'images' hoặc 'labels'.")
    exit()

image_files = set(os.path.splitext(f)[0] for f in os.listdir(images_dir))
label_files = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir))

total_images = len(image_files)
total_labels = len(label_files)
print(f"Tổng số file ảnh tìm thấy: {total_images}")
print(f"Tổng số file nhãn tìm thấy: {total_labels}")
print("-" * 50)


# --- KIỂM TRA 1: TÊN FILE CÓ KHỚP NHAU KHÔNG? ---
print(">>> KIỂM TRA 1: SỰ KHỚP NHAU CỦA TÊN FILE <<<")
matching_files = image_files.intersection(label_files)
images_without_labels = image_files.difference(label_files)
labels_without_images = label_files.difference(image_files)

print(f"Số lượng file có cả ảnh và nhãn khớp tên: {len(matching_files)}")

if images_without_labels:
    print(f"\n[CẢNH BÁO] {len(images_without_labels)} file ảnh sau đây KHÔNG có file nhãn tương ứng:")
    for filename in list(images_without_labels)[:5]: # Chỉ in ra 5 cái đầu tiên
        print(f"  - {filename}")

if labels_without_images:
    print(f"\n[CẢNH BÁO] {len(labels_without_images)} file nhãn sau đây KHÔNG có file ảnh tương ứng:")
    for filename in list(labels_without_images)[:5]:
        print(f"  - {filename}")

print("-" * 50)


# --- KIỂM TRA 2: NỘI DUNG FILE NHÃN CÓ HỢP LỆ KHÔNG? ---
print(">>> KIỂM TRA 2: NỘI DUNG CÁC FILE NHÃN <<<")
invalid_label_files = []
valid_files_count = 0

for filename in matching_files:
    label_path = os.path.join(labels_dir, f"{filename}.txt")
    is_valid = True
    
    try:
        with open(label_path, 'r') as f:
            # Kiểm tra file có rỗng không
            if os.path.getsize(label_path) == 0:
                invalid_label_files.append((filename, "File rỗng"))
                is_valid = False
                continue

            lines = f.readlines()
            if not lines:
                invalid_label_files.append((filename, "File rỗng (không có dòng nào)"))
                is_valid = False
                continue
                
            for i, line in enumerate(lines):
                parts = line.strip().split()
                
                # Phải có đúng 5 phần
                if len(parts) != 5:
                    invalid_label_files.append((filename, f"Dòng {i+1} không có 5 phần"))
                    is_valid = False
                    break
                
                # Kiểm tra các phần có phải là số không
                class_id_str, x_str, y_str, w_str, h_str = parts
                
                if not class_id_str.isdigit():
                    invalid_label_files.append((filename, f"Dòng {i+1}: class_id không phải số nguyên"))
                    is_valid = False
                    break
                
                class_id = int(class_id_str)
                if class_id != 0: # Giả sử bạn chỉ có 1 lớp là 0
                     invalid_label_files.append((filename, f"Dòng {i+1}: class_id ({class_id}) không phải 0"))
                     is_valid = False
                     break

                # Kiểm tra 4 tọa độ
                for coord_str in [x_str, y_str, w_str, h_str]:
                    try:
                        coord = float(coord_str)
                        if not (0.0 <= coord <= 1.0):
                            invalid_label_files.append((filename, f"Dòng {i+1}: Tọa độ {coord} nằm ngoài khoảng [0, 1]"))
                            is_valid = False
                            break
                    except ValueError:
                        invalid_label_files.append((filename, f"Dòng {i+1}: Tọa độ '{coord_str}' không phải là số"))
                        is_valid = False
                        break
                if not is_valid: break
    except Exception as e:
        invalid_label_files.append((filename, f"Lỗi không xác định: {e}"))
        is_valid = False

    if is_valid:
        valid_files_count += 1


print(f"Số lượng file có nhãn hợp lệ (định dạng đúng): {valid_files_count}")

if invalid_label_files:
    print(f"\n[LỖI] Tìm thấy {len(invalid_label_files)} file nhãn có nội dung KHÔNG HỢP LỆ:")
    for filename, reason in invalid_label_files[:10]: # Chỉ in 10 lỗi đầu tiên
        print(f"  - File: {filename}.txt, Lý do: {reason}")
        
print("-" * 50)
print(">>> KIỂM TRA HOÀN TẤT <<<")

if valid_files_count == 0:
    print("\n[KẾT LUẬN] KHÔNG có file nào hợp lệ để huấn luyện. Đây là lý do gây ra lỗi 'No valid images found'.")
    print("Hãy sửa các lỗi được liệt kê ở trên trong thư mục 'dataset' gốc của bạn.")
else:
     print(f"\n[KẾT LUẬN] Có {valid_files_count} file hợp lệ để huấn luyện. Nếu vẫn gặp lỗi, vấn đề có thể nằm ở đường dẫn trong file yaml.")