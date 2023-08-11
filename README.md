# [Ứng dụng ChatBot - Project xử lý ngôn ngữ tự nhiên - K32 Khoa học dữ liệu]


**Ứng dụng chatbot sử dụng OpenAI API**

---

## Nội dung chính
** Học viên: Trần Hoàng Vũ (22C01027) **
** Học viên: Bùi Tất Hiệp (22C01007) **

* Hiểu và trả lời bất kỳ văn bản, promt được truyền vào
* Lưu trữ tin nhắn của user nhập trong cơ sở dữ liệu và ghi nhớ bối cảnh đối thoại của chatbot
* Giao diện người dùng đáp ứng
* Chuẩn hóa API, giao diện Frontend có thể customize lại



**Below**: *Screenshot of the project*

![Screenshot][image]



## Requirements

* Python 3.6+
* Django 3.1+
* djangorestframework 3.12+
* python-dotenv
* openai

I **highly recommend** Sử dụng phiên bản mới nhất của các ứng dụng

## Installation

--> Thực hiện clone source với đường dẫn sau (dùng git để thực hiện, hoặc tải trực tiếp xuống):

    git clone https://github.com/TranHoangVukhtn/chatbot/
    
--> Di chuyển vào thư mục nơi chứa prject tệp dự án :

    cd chatbot
    
--> Tạo enviroment cho project :
```bash
# tạo vùng virtualenv
pip install virtualenv

# Sau đó tạo vùng enviroment 
virtualenv env #or python -m virtualenv if you're using Windows

```

--> Activate the virtual environment :
```bash
env\scripts\activate #or env\Scripts\activate.bat (Nếu dùng windows)

```

--> Install the requirements :
```bash
pip install -r requirements.txt

```

## Cấu hình Database

For example:
```
SECRET_KEY = 'some-secret-key' 
OPENAI_API_KEY = 'YOUR-API-KEY' # OPENAPI Key
https://platform.openai.com/account/api-keys

#DATABASES
ENGINE   = 'django.db.backends.sqlite3' 
NAME     = 'chat.db'

```
*Để được giải thích chi tiết về cách kết nối với cơ sở dữ liệu SQL cụ thể, hãy truy cập [Django documentation][django-docs]* 


--> Thực hiện trên terminal câu lệnh migrations xuống database:
```bash
python manage.py makemigrations
python manage.py migrate

```

--> Tạo superuser trên admin:

    python manage.py createsuperuser
    


#

## Running development server

--> Để run project ChatBot, thực hiện :
```bash
python manage.py runserver

```

> ⚠ Application sẽ start với http://127.0.0.1:8000/




## * Sử dụng docker container:*
Cài đặt docker engine với đường dẫn sau:https://www.docker.com

1. Docker images (chatbot-master_nlp)

2. docker run -p 8000:8000 chatbot-master_nlp

