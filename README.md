# Spanish Dialogue Practice App

Приложение для тренировки испанского языка через диалоги. Пользователь вводит сообщение, а модель Hugging Face на основе GPT-2 генерирует ответ на испанском языке.

## 🚀 Функциональность
- Ввод текста на испанском языке.
- Генерация контекстуальных ответов модели.
- Простая веб-страница для взаимодействия с приложением.

## 🛠️ Установка
1. Клонируйте репозиторий:
   ```bash
   git clone git@github.com:potashka/spanish.git
   cd spanish-dialogue-app

2. Установите зависимости:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Для Linux/Mac
    venv\Scripts\activate     # Для Windows
    pip install -r requirements.txt

3. Если возникнут проблемы с PyTorch установите PyTorch (выберите подходящую команду на официальном сайте):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


▶️ Запуск

Запустите сервер

    ```bash
    uvicorn app.main:app --reload
    Перейдите в браузере по адресу: http://127.0.0.1:8000
    ```

📝 Зависимости

 FastAPI — для разработки веб-приложения.
 Uvicorn — ASGI-сервер для запуска FastAPI.
 Transformers — библиотека Hugging Face для загрузки модели.
 PyTorch — для работы с моделями глубокого обучения.