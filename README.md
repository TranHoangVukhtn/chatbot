# [ChatBot]


**Ứng dụng chatbot sử dụng OpenAI API**

---

## Tổng quan




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

--> First of all, clone this repository:

    git clone https://github.com/TranHoangVukhtn/chatbot/
    
--> Move into the directory where we have the project files :

    cd chatbot
    
--> Create a virtual environment :
```bash
# Let's install virtualenv first
pip install virtualenv

# Then we create our virtual environment
virtualenv env #or python -m virtualenv if you're using Windows

```

--> Activate the virtual environment :
```bash
env\scripts\activate #or env\Scripts\activate.bat if you're using Windows

```

--> Install the requirements :
```bash
pip install -r requirements.txt

```

## Database configuration

--> Change name of the chatbot/.env.sample file to chatbot/.env and specify data of your database. For example:
```
SECRET_KEY = 'some-secret-key' 
OPENAI_API_KEY = 'YOUR-API-KEY' #specify your OpenAI API key, that you can get on https://platform.openai.com/account/api-keys

#DATABASES
ENGINE   = 'django.db.backends.sqlite3' 
NAME     = 'chat.db'

```
*For detailed explanation of how to connect to specific SQL database visit [Django documentation][django-docs]* 


--> Apply migrations to your database:
```bash
python manage.py makemigrations
python manage.py migrate

```

--> Create superuser:

    python manage.py createsuperuser
    


#

## Running development server

--> To run the ChatBot, use :
```bash
python manage.py runserver

```

> ⚠ Then, the development server will be started at http://127.0.0.1:8000/




## * Sử dụng docker container:*

