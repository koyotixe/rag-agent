import sounddevice as sd
import torch
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

executor = ThreadPoolExecutor(max_workers=1)
_model = None
_device = torch.device('cpu')

def get_model():
    global _model
    if _model is None:
        _model, _ = torch.hub.load(
            'snakers4/silero-models',
            'silero_tts',
            language='ru',
            speaker='v4_ru'
        )
        _model.to(_device)
    return _model

def play_speech_sync(text: str, speaker: str = 'baya', sample_rate: int = 24000):
    try:
        model = get_model()
        audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
        sd.play(audio.numpy(), sample_rate)
        sd.wait()
        return True
    except Exception as e:
        return False

async def play_speech_async(text: str, speaker: str = 'baya', sample_rate: int = 24000):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        play_speech_sync,
        text,
        speaker,
        sample_rate
    )

def get_audio_array(text: str, speaker: str = 'baya', sample_rate: int = 24000):
    try:
        model = get_model()
        audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
        return audio.numpy(), sample_rate
    except Exception as e:
        return None, None