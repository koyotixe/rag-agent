from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import random
import asyncio
from datetime import datetime
import warnings
import base64
import io
import os

from tts_silero import play_speech_async

from rag_system import (
    get_rag_answer, 
    is_rag_ready, 
    load_existing_index,
    get_documents_list
)

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

warnings.filterwarnings("ignore")

def detect_emotion(text):
    text_lower = text.lower()
    
    happy_words = ['рад', 'счастье', 'отлично', 'круто', 'love', 'good', 'great', 'happy']
    sad_words = ['грус', 'печал', 'плохо', 'жаль', 'bad', 'sad', 'cry']
    angry_words = ['зол', 'гнев', 'бесит', 'надоел', 'angry', 'mad', 'hate']
    surprise_words = ['вау', 'ого', 'неожидан', 'surprise', 'wow', 'oh']
    
    happy_count = sum(1 for word in happy_words if word in text_lower)
    sad_count = sum(1 for word in sad_words if word in text_lower)
    angry_count = sum(1 for word in angry_words if word in text_lower)
    surprise_count = sum(1 for word in surprise_words if word in text_lower)
    
    emotions = {
        'happy': happy_count,
        'sad': sad_count,
        'angry': angry_count,
        'surprised': surprise_count,
        'calm': 0.5
    }
    
    max_emotion = max(emotions, key=emotions.get)
    max_value = emotions[max_emotion]
    
    if max_value == 0 and max_emotion == 'calm':
        return 'calm', 0.3
    
    intensity = min(0.3 + max_value * 0.3, 1.0)
    return max_emotion, intensity

async def generate_response(text):
    text_lower = text.lower().strip()
    
    if text_lower in ['привет', 'здравствуй', 'здравствуйте', 'приветик', 'салют', 'hello', 'hi']:
        return "Здравствуйте! Я Ева, ваш AI-помощник. Я могу отвечать на вопросы по лекциям по нейронным сетям. Спрашивайте!"
    
    if any(word in text_lower for word in ['кто ты', 'представься', 'как тебя зовут', 'твое имя', 'расскажи о себе', 'ты кто']):
        return "Меня зовут Ева. Я AI-ассистент, созданный для помощи в изучении нейронных сетей. Я умею отвечать на вопросы по лекциям, распознавать речь и говорить голосом. Чем могу помочь?"
    
    if any(word in text_lower for word in ['как дела', 'как ты', 'как настроение', 'как жизнь']):
        return "У меня всё отлично! Я готова отвечать на ваши вопросы по лекциям. Что хотите узнать?"
    
    if any(word in text_lower for word in ['спасибо', 'благодарю', 'thanks']):
        return "Пожалуйста! Я всегда рада помочь. Обращайтесь, если будут вопросы по лекциям."
    
    if any(word in text_lower for word in ['пока', 'до свидания', 'bye', 'goodbye']):
        return "До свидания! Буду рада помочь снова. Удачи в изучении нейронных сетей!"
    
    if is_rag_ready():
        return await get_rag_answer(text)
    else:
        responses = [
            f"Интересно, ты сказал: '{text}'. Расскажи подробнее!",
            f"Я слышу тебя. {text} — это важно.",
            "Документ пока не загружен. Положите файлы в папку backend/rag_documents",
            "Я Ева, ваш помощник по лекциям. Спрашивайте!"
        ]
        return random.choice(responses)

def transcribe_audio(audio_base64: str):
    if not SPEECH_RECOGNITION_AVAILABLE:
        return None
    
    try:
        audio_data = base64.b64decode(audio_base64)
        
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="ru-RU")
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return text
        
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None
    except Exception as e:
        return None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections = {}
tts_queue = asyncio.Queue()

async def tts_worker():
    while True:
        try:
            text, client_id = await tts_queue.get()
            await play_speech_async(text, speaker='baya')
            tts_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            pass

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(tts_worker())
    load_existing_index()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "text":
                user_text = message["text"]
                
                await websocket.send_json({"type": "status", "status": "thinking"})
                
                response_text = await generate_response(user_text)
                emotion, intensity = detect_emotion(user_text)
                
                visemes = []
                duration_per_char = 0.2
                current_time = 0
                
                for char in response_text.lower():
                    if char in 'аояёуюиыэе':
                        visemes.append({"viseme": "A", "start": current_time})
                    elif char in 'бвгджзйклмнпрстфхцчшщ':
                        visemes.append({"viseme": "M", "start": current_time})
                    else:
                        visemes.append({"viseme": "rest", "start": current_time})
                    current_time += duration_per_char
                
                await websocket.send_json({
                    "type": "speak",
                    "text": response_text,
                    "emotion": emotion,
                    "intensity": intensity,
                    "visemes": visemes
                })
                
                await tts_queue.put((response_text, client_id))
            
            elif message["type"] == "audio":
                audio_base64 = message["audio"]
                
                if SPEECH_RECOGNITION_AVAILABLE:
                    recognized_text = transcribe_audio(audio_base64)
                    
                    if recognized_text:
                        await websocket.send_json({"type": "status", "status": "thinking"})
                        
                        response_text = await generate_response(recognized_text)
                        emotion, intensity = detect_emotion(recognized_text)
                        
                        visemes = []
                        duration_per_char = 0.2
                        current_time = 0
                        
                        for char in response_text.lower():
                            if char in 'аояёуюиыэе':
                                visemes.append({"viseme": "A", "start": current_time})
                            elif char in 'бвгджзйклмнпрстфхцчшщ':
                                visemes.append({"viseme": "M", "start": current_time})
                            else:
                                visemes.append({"viseme": "rest", "start": current_time})
                            current_time += duration_per_char
                        
                        await websocket.send_json({
                            "type": "speak",
                            "text": response_text,
                            "emotion": emotion,
                            "intensity": intensity,
                            "visemes": visemes
                        })
                        
                        await tts_queue.put((response_text, client_id))
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Не удалось распознать речь. Попробуйте ещё раз."
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Распознавание речи не доступно. Установите SpeechRecognition."
                    })
            
            elif message["type"] == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
    
    except Exception as e:
        pass
    finally:
        if client_id in active_connections:
            del active_connections[client_id]

@app.get("/")
async def root():
    return {
        "message": "AI Avatar Server with RAG + TTS + SpeechRecognition", 
        "status": "running", 
        "connections": len(active_connections),
        "tts_ready": True,
        "speech_recognition_ready": SPEECH_RECOGNITION_AVAILABLE,
        "rag_ready": is_rag_ready()
    }

@app.get("/test-tts")
async def test_tts(text: str = "Привет! Это тест голоса."):
    await play_speech_async(text, speaker='baya')
    return {"status": "speaking", "text": text}

@app.get("/rag_status")
async def rag_status():
    return {
        "rag_ready": is_rag_ready(),
        "documents": get_documents_list() if is_rag_ready() else []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)