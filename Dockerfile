FROM python:3.9
WORKDIR /app

ADD requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install --upgrade -r requirements.txt

EXPOSE 8888

COPY ./ /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "4", "--proxy-headers"]