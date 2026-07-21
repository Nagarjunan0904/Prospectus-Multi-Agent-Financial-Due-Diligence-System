FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Install CPU-only torch explicitly BEFORE the rest of requirements.txt,
# same reasoning as local dev: the default torch wheel pulls CUDA
# binaries that are both huge and useless without a GPU.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["supervisord", "-c", "supervisord.conf"]
