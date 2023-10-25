# Phân Loại Ảnh Các Loài Mèo Lớn bằng SVM và Decision Tree

## Giới Thiệu

Dự án này tập trung vào việc phân loại hình ảnh của các loài mèo lớn, bao gồm Sư tử, Hổ, Báo, và nhiều loài mèo lớn khác, bằng cách sử dụng hai thuật toán phân loại khác nhau: Support Vector Machine (SVM) và Decision Tree. Mục tiêu chính là xây dựng và huấn luyện các mô hình phân loại cho việc xác định loài mèo lớn trong hình ảnh dựa trên dữ liệu từ các tệp JSON chứa thông tin về hình ảnh và nhãn tương ứng.

## Thư Viện và Thuật Toán Sử Dụng

Dự án này sử dụng các thư viện và thuật toán sau:

### Thư Viện

1. `json`: Sử dụng để xử lý tệp JSON chứa thông tin hình ảnh và nhãn.
2. `pandas`: Sử dụng để làm việc với dữ liệu trong dạng DataFrame.
3. `scikit-learn`: Sử dụng để xây dựng và đánh giá mô hình SVM và Decision Tree.
4. `joblib`: Sử dụng để lưu trữ và nạp lại các mô hình đã huấn luyện.
5. `time`: Sử dụng để đo thời gian huấn luyện mô hình.
6. `matplotlib`: Sử dụng để vẽ biểu đồ hiển thị giá trị hàm mất mát của các mô hình.
7. `tabulate`: Sử dụng để tạo bảng thông tin đánh giá mô hình.

### Thuật Toán

1. **Support Vector Machine (SVM)**: SVM là một thuật toán học máy sử dụng trong việc phân loại và hồi quy. Trong dự án này, SVM được áp dụng để phân loại ảnh các loài mèo lớn.

2. **Decision Tree**: Decision Tree là một loại thuật toán học máy sử dụng cấu trúc cây quyết định để phân loại dữ liệu. Decision Tree cũng được sử dụng để phân loại ảnh các loài mèo lớn và so sánh với SVM.

## Bắt Đầu

Để bắt đầu sử dụng dự án, bạn có thể thực hiện các bước sau:

1. Clone dự án từ GitHub:

```bash
git clone https://github.com/Zavield-TheChosen1/Classify-images-of-Large-Cats-using-SVM-and-ID3-algorithms.git
cd Classify-images-of-Large-Cats-using-SVM-and-ID3-algorithms
Cài đặt các thư viện cần thiết:
bash
Copy code
pip install -r requirements.txt
Tài Liệu Tham Khảo
Dưới đây là một số tài liệu tham khảo về thuật toán và thư viện được sử dụng trong dự án:

Thuật Toán
Support Vector Machine (SVM):

Hướng dẫn scikit-learn về SVM
Decision Tree:

Hướng dẫn scikit-learn về Decision Trees
Thư Viện
scikit-learn:

Tài liệu chính thức của scikit-learn
Pandas:

Tài liệu chính thức của pandas
Thông Tin Liên Hệ
Email: haanhty2711@gmail.com
Thành Viên Phát Triển
Hà Anh Tú
Nguyễn Thị Thanh Thảo
Nguyễn Tiến Dũng
Vũ Văn Hoàng
Hướng Phát Triển Sắp Tới
Các hướng phát triển sắp tới bao gồm:

Tối ưu hóa hiệu suất mô hình và tăng độ chính xác.

Mở rộng dự án để phân loại nhiều loài mèo khác nhau.

Phát triển giao diện người dùng (UI) để tạo trải nghiệm sử dụng dễ dàng hơn.

