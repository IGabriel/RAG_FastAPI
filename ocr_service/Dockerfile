FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Allow switching apt mirrors for restricted networks (e.g. mainland China).
# Use full URLs (with scheme), e.g.
#   APT_MIRROR_URL=http://mirrors.aliyun.com/debian
#   APT_SECURITY_URL=http://mirrors.aliyun.com/debian-security
ARG APT_MIRROR_URL=http://deb.debian.org/debian
ARG APT_SECURITY_URL=http://security.debian.org/debian-security

# Tesseract + Chinese language pack + PDF rendering (poppler)
RUN set -eux; \
    sed -i "s|http://deb.debian.org/debian|${APT_MIRROR_URL}|g" /etc/apt/sources.list; \
    sed -i "s|http://security.debian.org/debian-security|${APT_SECURITY_URL}|g" /etc/apt/sources.list; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ocr_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ocr_service/app /app/app

EXPOSE 9001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001"]
