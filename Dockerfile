FROM python:3.10-slim

WORKDIR /app

# Now copy the rest of the app
COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

COPY models/label_encoder.pkl /app/models/label_encoder.pkl

# Download nltk data
RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

#local run
CMD ["python", "app.py"]  

#For production with gunicorn:
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "app:app"]