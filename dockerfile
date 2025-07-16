# Sử dụng image Python
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết
COPY . .

RUN pip install --no-cache-dir numpy==1.26.4

RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Cài đặt cron
RUN apt-get update && apt-get install -y cron

# Cấu hình múi giờ UTC+7
RUN ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Cài đặt thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép file crontab vào hệ thống
COPY crontab /etc/cron.d/sentiment_project_cron

# Cấp quyền và chạy
RUN chmod 0644 /etc/cron.d/sentiment_project_cron && crontab /etc/cron.d/sentiment_project_cron

RUN chmod +x /app/pipeline.sh

# Tạo file log
RUN touch /var/log/cron.log /var/log/pipeline.log

# Mở cổng 8000 cho API
EXPOSE 8000

# Khởi động cả cron và FastAPI khi container chạy
CMD ["sh", "-c", "cron && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
