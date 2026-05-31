# Eva AI Assistant

Eva AI Assistant — AI-ассистент с голосовым интерфейсом и системой Retrieval-Augmented Generation (RAG) для работы с лекциями и обучающими материалами.

## Возможности

* Ответы на вопросы по лекциям и документам
* Загрузка PDF и TXT файлов
* Semantic search по материалам
* Voice assistant functionality
* Speech-to-Text распознавание речи
* Text-to-Speech синтез речи
* Real-time взаимодействие через WebSocket
* Определение эмоций пользователя
* Асинхронная обработка запросов

---

# Архитектура проекта

## RAG Pipeline

1. Документы загружаются в систему
2. Текст разбивается на чанки
3. Создаются embeddings
4. Embeddings сохраняются в FAISS
5. По запросу пользователя выполняется retrieval
6. LLM генерирует ответ на основе найденного контекста

## AI Assistant Pipeline

1. Пользователь отправляет текст или голос
2. Speech-to-Text преобразует аудио в текст
3. Выполняется semantic retrieval
4. LLM генерирует ответ
5. Определяется эмоция сообщения
6. Генерируется голосовой ответ
7. Ответ отправляется через WebSocket

---

# Используемые технологии

## Backend

* Python
* FastAPI
* WebSocket
* asyncio

## AI / ML

* LangChain
* FAISS
* HuggingFace Embeddings
* GigaChat API
* RAG (Retrieval-Augmented Generation)

## Voice

* SpeechRecognition
* Silero TTS

---

# Основные функции

## Работа с документами

* Загрузка PDF и TXT файлов
* Индексация документов
* Векторный поиск
* Persistent FAISS index

## AI Assistant

* Генерация ответов
* Контекстные ответы по лекциям
* Работа с несколькими документами
* Async request processing

## Voice Features

* Распознавание речи
* Голосовой вывод
* Lip-sync visemes generation
* Emotion detection

---

# Структура проекта

```bash id="5h2s91"
backend/
│
├── rag_system.py
├── server.py
├── tts_silero.py
│
├── rag_documents/
│
├── document_index/
│
└── requirements.txt
```

---

# Запуск проекта

## Установка зависимостей

```bash id="v31ks2"
pip install -r requirements.txt
```

## Запуск сервера

```bash id="gx91ks"
python server.py
```

Сервер будет доступен по адресу:

```bash id="kq91bz"
http://localhost:8000
```

# Особенности реализации

* Асинхронная архитектура на asyncio
* Real-time communication через WebSocket
* Semantic retrieval через FAISS
* MMR search для повышения качества retrieval
* Lazy initialization моделей
* Persistent vector storage
* Очередь обработки TTS

---

# Планы по развитию

* Поддержка DOCX
* История диалогов
* Memory system
* Улучшение AI-агентов
* Streaming responses
* Docker deployment
* Web UI
* Multi-user support
