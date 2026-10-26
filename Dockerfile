FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD streamlit run ui/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0
