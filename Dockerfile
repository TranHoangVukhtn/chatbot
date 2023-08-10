# Sử dụng một hình ảnh cơ sở Python
FROM python:3.8

# Đặt thư mục làm việc cho ứng dụng
WORKDIR /Users/tramy/Documents/Projects/chatbot-master

# Sao chép các tệp cấu hình và yêu cầu cài đặt vào thư mục làm việc
COPY requirements.txt .

# Cài đặt các yêu cầu
RUN pip install -r requirements.txt

# Sao chép tất cả các tệp từ thư mục hiện tại vào thư mục làm việc
COPY /Users/tramy/Documents/Projects/chatbot-master

# Mở cổng cho ứng dụng Django
EXPOSE 8000

# Khởi chạy dịch vụ Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

