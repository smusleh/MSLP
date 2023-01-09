FROM python:3.10
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY . /app
EXPOSE 8000
WORKDIR /app
CMD uvicorn localization_api:app --host 0.0.0.0 --reload