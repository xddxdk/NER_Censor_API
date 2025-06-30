# Базовый образ
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y git

# Рабочая директория в контейнере
WORKDIR /app

# Копируем зависимости и устанавливаем
COPY api/requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Копируем api.py
COPY api/api.py ./

# Копируем модель (она в ../model) в папку /app/model
COPY ../model ./model

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]