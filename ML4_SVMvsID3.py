#Import các thư viện cần thiết
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

# Thời điểm bắt đầu huấn luyện
thời_điểm_bắt_đầu = time.time()

# Địa chỉ lưu trữ các file JSON
json_files = [
    "E:/Code/ML4_SVMvsID3/anh_nhan_dan_1.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_2.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_3.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_6.json",
    "E:/Code/ML4_SVMvsID3/anh_dan_nhan_4.json"
]

# Khởi tạo danh sách để lưu trữ giá trị hàm mất mát qua các lượt huấn luyện
loss_values_svm = []  # Danh sách giá trị hàm mất mát cho SVM
loss_values_decision_tree = []  # Danh sách giá trị hàm mất mát cho Decision Tree

# Khởi tạo một danh sách trống để lưu trữ dữ liệu hình ảnh từ tất cả các tệp
tất_cả_dữ_liệu_hình_ảnh = []

for json_file in json_files:
    if os.path.exists(json_file):  # Kiểm tra xem tệp có tồn tại không
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Kiểm tra xem dữ liệu hình ảnh tồn tại và có định dạng đúng không
        if "_via_img_metadata" in data and isinstance(data["_via_img_metadata"], dict):
            print(f"Dữ liệu hình ảnh trong tệp {json_file} hợp lệ.")
        else:
            print(f"Dữ liệu hình ảnh trong tệp {json_file} không hợp lệ.")
    else:
        print(f"Tệp {json_file} không tồn tại.")

# Khởi tạo một danh sách trống để lưu trữ dữ liệu hình ảnh từ tất cả các tệp
tất_cả_dữ_liệu_hình_ảnh = []

for json_file in json_files:
    # Đọc tệp JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Kiểm tra thông tin cần thiết trong dữ liệu hình ảnh
    if "_via_img_metadata" in data and isinstance(data["_via_img_metadata"], dict):
        for image_id, image_info in data["_via_img_metadata"].items():
            # Kiểm tra thông tin quan trọng (thay thế bằng thông tin cần kiểm tra)
            if "regions" in image_info and isinstance(image_info["regions"], list):
                print(f"Đã tìm thấy thông tin vùng (regions) hợp lệ trong tệp {json_file}.")
            else:
                print(f"Không tìm thấy thông tin vùng (regions) hợp lệ trong tệp {json_file}.")

    # Khởi tạo một danh sách để lưu trữ dữ liệu hình ảnh cho tệp này
    dữ_liệu_hình_ảnh = []

    # Trích xuất thông tin hình ảnh từ tệp JSON
    for image_id, image_info in data["_via_img_metadata"].items():
        filename = image_info["filename"]
        size = image_info["size"]

        # Kiểm tra xem có vùng (regions) không
        if "regions" in image_info:
            regions = image_info["regions"]
            if len(regions) > 0:
                # Trích xuất region_attributes từ region đầu tiên (giả sử chỉ có một region)
                region_attributes = regions[0].get("region_attributes", {})
                name = region_attributes.get("Name", "Không_xác_định")
                type_ = region_attributes.get("Species", "Không_xác_định")
                identifying = region_attributes.get("Identifying", "Không_xác_định")
            else:
                name = "Không_xác_định"
                type_ = "Không_xác_định"
                identifying = "Không_xác_định"
        else:
            name = "Không_xác_định"
            type_ = "Không_xác_định"
            identifying = "Không_xác_định"

        # Thêm thông tin vào danh sách
        dữ_liệu_hình_ảnh.append({
            "filename": filename,
            "size": size,
            "name": name,
            "type": type_,
            "identifying": identifying
        })

    # Tạo một DataFrame từ danh sách dữ liệu hình ảnh cho tệp này
    df = pd.DataFrame(dữ_liệu_hình_ảnh)

    # Sử dụng LabelEncoder để chuyển đổi các cột chuỗi thành số
    label_encoder = LabelEncoder()
    các_cột_chuỗi = ["name", "type", "identifying"]

    for col in các_cột_chuỗi:
        df[col] = label_encoder.fit_transform(df[col])

    # Loại bỏ cột "filename" khỏi dữ liệu huấn luyện
    df.drop("filename", axis=1, inplace=True)

    # Thêm dữ liệu cho tệp này vào dữ liệu tổng cộng
    tất_cả_dữ_liệu_hình_ảnh.append(df)

# Kết hợp dữ liệu từ tất cả các tệp
dữ_liệu_kết_hợp = pd.concat(tất_cả_dữ_liệu_hình_ảnh, ignore_index=True)

# Tạo tập dữ liệu huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(dữ_liệu_kết_hợp.drop("type", axis=1), dữ_liệu_kết_hợp["type"], test_size=0.2)

# Kiểm tra kích thước của tập huấn luyện và tập kiểm tra
kích_thước_tập_huấn_luyện = len(x_train)
kích_thước_tập_kiểm_tra = len(x_test)

## Phần mã cho mô hình SVM

# Tạo mô hình SVM
svm_model = SVC()

# Huấn luyện mô hình SVM
svm_model.fit(x_train, y_train)

# So sánh tỷ lệ tập kiểm tra thực tế với tỷ lệ bạn mong đợi
tỉ_lệ_tập_kiểm_tra_thực_tế = kích_thước_tập_kiểm_tra / (kích_thước_tập_huấn_luyện + kích_thước_tập_kiểm_tra)

if tỉ_lệ_tập_kiểm_tra_thực_tế == 0.2:
    print("Việc tách tập huấn luyện và tập kiểm tra đã được thực hiện đúng cách và tỷ lệ tập kiểm tra là 0.2.")
else:
    print("Có lỗi trong việc tách tập huấn luyện và tập kiểm tra, hoặc tỷ lệ tập kiểm tra không phù hợp.")

# Dự đoán trên tập kiểm tra cho mô hình SVM
y_pred_svm = svm_model.predict(x_test)

# Tính giá trị hàm mất mát cho mô hình SVM và thêm vào danh sách loss_values_svm
current_loss_svm = svm_model.score(x_test, y_test)
loss_values_svm.append(current_loss_svm)

# Tính độ chính xác cho mô hình SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Độ chính xác của mô hình SVM:", accuracy_svm)

# Tính độ nhạy và độ cụ thể cho mô hình SVM
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
print("Độ nhạy (Recall) của mô hình SVM:", recall_svm)
print("Độ cụ thể (Precision) của mô hình SVM:", precision_svm)

# Tính ma trận nhầm lẫn cho mô hình SVM
confusion_svm = confusion_matrix(y_test, y_pred_svm)
print("Ma trận nhầm lẫn của mô hình SVM:\n", confusion_svm)

# Lưu mô hình SVM đã huấn luyện vào một tệp
đường_dẫn_mô_hình_svm = 'mô_hình_svm_đã_huấn_luyện.joblib'
joblib.dump(svm_model, đường_dẫn_mô_hình_svm)

## Phần mã cho mô hình Decision Tree (ID3)

# Tạo mô hình Decision Tree
decision_tree_model = DecisionTreeClassifier()

# Huấn luyện mô hình Decision Tree
decision_tree_model.fit(x_train, y_train)

# So sánh tỷ lệ tập kiểm tra thực tế với tỷ lệ bạn mong đợi
tỉ_lệ_tập_kiểm_tra_thực_tế = kích_thước_tập_kiểm_tra / (kích_thước_tập_huấn_luyện + kích_thước_tập_kiểm_tra)

if tỉ_lệ_tập_kiểm_tra_thực_tế == 0.2:
    print("Việc tách tập huấn luyện và tập kiểm tra đã được thực hiện đúng cách và tỷ lệ tập kiểm tra là 0.2.")
else:
    print("Có lỗi trong việc tách tập huấn luyện và tập kiểm tra, hoặc tỷ lệ tập kiểm tra không phù hợp.")

# Dự đoán trên tập kiểm tra cho mô hình Decision Tree
y_pred_decision_tree = decision_tree_model.predict(x_test)

# Tính giá trị hàm mất mát cho mô hình Decision Tree và thêm vào danh sách loss_values_decision_tree
current_loss_decision_tree = decision_tree_model.score(x_test, y_test)
loss_values_decision_tree.append(current_loss_decision_tree)

# Tính độ chính xác cho mô hình Decision Tree
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print("Độ chính xác của mô hình Decision Tree:", accuracy_decision_tree)

# Tính độ nhạy và độ cụ thể cho mô hình Decision Tree
precision_decision_tree = precision_score(y_test, y_pred_decision_tree, average='macro')
recall_decision_tree = recall_score(y_test, y_pred_decision_tree, average='macro')
print("Độ nhạy (Recall) của mô hình Decision Tree:", recall_decision_tree)
print("Độ cụ thể (Precision) của mô hình Decision Tree:", precision_decision_tree)

# Tính ma trận nhầm lẫn cho mô hình Decision Tree
confusion_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
print("Ma trận nhầm lẫn của mô hình Decision Tree:\n", confusion_decision_tree)

# Lưu mô hình Decision Tree đã huấn luyện vào một tệp
đường_dẫn_mô_hình_decision_tree = 'mô_hình_decision_tree_đã_huấn_luyện.joblib'
joblib.dump(decision_tree_model, đường_dẫn_mô_hình_decision_tree)

# Thời điểm kết thúc huấn luyện
thời_điểm_kết_thúc = time.time()

# Thời gian huấn luyện cho SVM (tính bằng giây)
thời_gian_huấn_luyện_svm = thời_điểm_kết_thúc - thời_điểm_bắt_đầu

# Thời gian huấn luyện cho Decision Tree (tính bằng giây)
thời_gian_huấn_luyện_decision_tree = thời_điểm_kết_thúc - thời_điểm_bắt_đầu

# Đánh giá mô hình SVM (tạo bảng)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='macro')
svm_recall = recall_score(y_test, y_pred_svm, average='macro')
svm_confusion = confusion_matrix(y_test, y_pred_svm)

# Đánh giá mô hình Decision Tree (tạo bảng)
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
decision_tree_precision = precision_score(y_test, y_pred_decision_tree, average='macro')
decision_tree_recall = recall_score(y_test, y_pred_decision_tree, average='macro')
decision_tree_confusion = confusion_matrix(y_test, y_pred_decision_tree)

# Tạo biểu đồ đánh giá mô hình SVM và ID3
def plot_evaluation(title, accuracy, precision, recall, confusion):
    plt.figure(figsize=(12, 4))
    
    # Đánh giá độ chính xác, độ nhạy, độ cụ thể
    plt.subplot(131)
    plt.bar(['Accuracy', 'Precision', 'Recall'], [accuracy, precision, recall])
    plt.title(f'{title} Evaluation Metrics')
    plt.ylim([0, 1])
    
    # Hiển thị ma trận nhầm lẫn
    plt.subplot(132)
    plt.imshow(confusion, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(f'{title} Confusion Matrix')
    plt.colorbar()
    
    # Hiển thị các nhãn trên ma trận nhầm lẫn
    num_classes = len(confusion)
    plt.xticks(np.arange(num_classes), np.arange(1, num_classes + 1), rotation=45)
    plt.yticks(np.arange(num_classes), np.arange(1, num_classes + 1))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ đánh giá mô hình SVM
plot_evaluation('SVM', svm_accuracy, svm_precision, svm_recall, svm_confusion)

# Vẽ biểu đồ đánh giá mô hình Decision Tree
plot_evaluation('Decision Tree', decision_tree_accuracy, decision_tree_precision, decision_tree_recall, decision_tree_confusion)


# Đánh giá mô hình SVM
điểm_svm = svm_model.score(x_test, y_test)
# Tạo bảng thông tin mô hình SVM
svm_info = [
    ["Độ chính xác", f"{điểm_svm:.2%}"],
    ["Độ nhạy (Recall)", f"{recall_svm:.2f}"],
    ["Độ cụ thể (Precision)", f"{precision_svm:.2f}"],
    ["Thời gian huấn luyện", f"{thời_gian_huấn_luyện_svm:.2f} giây"],
]
print("\nThông tin mô hình SVM:")
print(tabulate(svm_info, headers=["Đặc điểm", "Giá trị"], tablefmt="fancy_grid"))

# Đánh giá mô hình Decision Tree
điểm_decision_tree = decision_tree_model.score(x_test, y_test)
# Tạo bảng thông tin mô hình Decision Tree
decision_tree_info = [
    ["Độ chính xác", f"{điểm_decision_tree:.2%}"],
    ["Độ nhạy (Recall)", f"{recall_decision_tree:.2f}"],
    ["Độ cụ thể (Precision)", f"{precision_decision_tree:.2f}"],
    ["Thời gian huấn luyện", f"{thời_gian_huấn_luyện_decision_tree:.2f} giây"],
]
print("\nThông tin mô hình Decision Tree:")
print(tabulate(decision_tree_info, headers=["Đặc điểm", "Giá trị"], tablefmt="fancy_grid"))

# In số lượng mẫu dữ liệu
số_lượng_mẫu = len(dữ_liệu_kết_hợp)
print(f"Số lượng mẫu dữ liệu: {số_lượng_mẫu}")

print(f"Thời gian huấn luyện cho SVM: {thời_gian_huấn_luyện_svm:.2f} giây")
print(f"Thời gian huấn luyện cho Decision Tree: {thời_gian_huấn_luyện_decision_tree:.2f} giây")

# Vẽ biểu đồ giá trị hàm mất mát cho SVM
plt.figure(figsize=(8, 6))
plt.plot(list(range(1, len(loss_values_svm) + 1)), loss_values_svm, marker='o', linestyle='-', color='b')
plt.title('Biểu đồ Hàm Mất Mát cho SVM qua Các Lượt Huấn Luyện')
plt.xlabel('Lượt Huấn Luyện')
plt.ylabel('Giá trị Hàm Mất Mát')
plt.grid(True)
plt.show()

# Vẽ biểu đồ giá trị hàm mất mát cho Decision Tree
plt.figure(figsize=(8, 6))
plt.plot(list(range(1, len(loss_values_decision_tree) + 1)), loss_values_decision_tree, marker='o', linestyle='-', color='g')
plt.title('Biểu đồ Hàm Mất Mát cho Decision Tree qua Các Lượt Huấn Luyện')
plt.xlabel('Lượt Huấn Luyện')
plt.ylabel('Giá trị Hàm Mất Mát')
plt.grid(True)
plt.show()

# Thông báo huấn luyện đã hoàn thành
print(f"Huấn luyện đã hoàn thành. Mô hình SVM đã được lưu tại: {đường_dẫn_mô_hình_svm}")
print(f"Mô hình Decision Tree đã được lưu tại: {đường_dẫn_mô_hình_decision_tree}")
