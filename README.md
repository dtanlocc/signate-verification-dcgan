# README.md

## Giới Thiệu

Đây là một Dự án này sử dụng Deep Convolutional Generative Adversarial Networks (DCGAN) từ PyTorch để phân biệt chữ ký thật và giả.

## Ý Tưởng Dự Án

Dự án này nhằm mục đích áp dụng DCGAN để tạo model dự đoán chính xác các chữ kí giả.
## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.x
- pip (công cụ quản lý gói cho Python)

### Hướng Dẫn Cài Đặt

1. **Clone repo từ GitHub:**

    \```sh
    git clone https://github.com/dtanlocc/signate-verification-dcgan.git
    cd repository
    \```

2. **Tạo virtual environment (môi trường ảo) và kích hoạt nó:**

    \```sh
    python -m venv env
    source env/bin/activate  # Trên Windows sử dụng: env\\\\Scripts\\\\activate
    \```

3. **Cài đặt các gói yêu cầu:**

    \```sh
    pip install -r requirements.txt
    \```
4. **Download dataset từ Kaggle:**

    Tập dữ liệu sử dụng trong dự án này có thể được tải xuống từ Kaggle: [Handwritten Signature Datasets](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets).

## Sử Dụng

Sau khi cài đặt các yêu cầu, bạn có thể chạy dự án bằng cách sử dụng lệnh sau:

\```sh
python main.py
\```

Thay đổi \`root\` và \'dataset_name\' bằng root và dataset_name cần thiết để phù hợp với dataset của bạn

## Kết Quả

Dưới đây là kết quả của dự án:

### Dataset CEDAR

![Kết quả minh họa](images\Figure_1.png)

*Hình 1: ROC của dataset*

![Kết quả minh họa](images\Figure_4.png)
*Hình 2: Metric của dataset*


### Dataset BHSig260-Hindi

![Kết quả minh họa](images\Figure_2.png)

*Hình 3: ROC của dataset*

![Kết quả minh họa](images\Figure_5.png)
*Hình 4: Metric của dataset*

### Dataset BHSig260-Bengali

![Kết quả minh họa](images\Figure_3.png)

*Hình 5: ROC của dataset*

![Kết quả minh họa](images\Figure_6.png)
*Hình 6: Metric của dataset*

## Tổng hợp kết quả

Nhìn chung thì kết quả cho ra rất tốt trên dataset CEDAR. Tuy trên 2 dataset còn lại thì kết quả không được tốt nhưng recall về xác thực chữ kí giả khá là tốt. Ở đây, tôi chưa áp dụng kĩ thuật data augmentation để tăng cường dữ liệu. Nên kết quả cho ra rất tốt. Trong tương lai tôi sẽ áp dụng tăng cương dữ liệu để thực hiện lại xem có thể tốt hơn không