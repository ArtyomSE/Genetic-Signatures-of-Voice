# 1. Берем официальный готовый образ от PyTorch с поддержкой CUDA
FROM ghcr.io/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Убираем интерактивные вопросы при сборке
ENV DEBIAN_FRONTEND=noninteractive

# 2. Доустанавливаем только аудио-декодер ffmpeg (он обязателен для Whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Копируем зависимости
COPY requirements.txt .

# Из requirements.txt нужно УДАЛИТЬ строки с torch и torchaudio, 
# так как они уже вшиты в этот образ! Оставьте там numpy, pandas, whisper, tqdm.
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем скрипт
COPY main.py .

# 5. Запуск (в этом образе по умолчанию настроен правильный Python)
CMD ["python", "main.py"]
