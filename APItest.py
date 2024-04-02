import sounddevice as sd
import numpy as np
import requests
import threading
from queue import Queue
from time import sleep
import io

# 设置采样率和录音时长
sample_rate = 16000  # Hz
duration = 1  # seconds, 调整录音时长为需要的秒数

# 创建一个队列来保存音频数据
audio_queue = Queue()


def audio_callback(indata, frames, time, status):
    """回调函数，用于捕获录音数据"""
    if status:
        print(status)
    audio_queue.put(indata.copy())


def transcribe_audio():
    """处理音频数据并实时转写"""
    global audio_queue
    accumulated_audio = np.array([], dtype='int16')  # 修改为整型，对应16位PCM编码
    while True:
        while not audio_queue.empty():
            data = audio_queue.get()
            # 确保音频数据是一维数组
            if data.ndim > 1:
                data = data.flatten()
            accumulated_audio = np.concatenate((accumulated_audio, data))
        if len(accumulated_audio) >= sample_rate * duration:
            # 使用BytesIO而不是临时文件
            audio_bytes = io.BytesIO()
            audio_bytes.write(accumulated_audio.tobytes())
            audio_bytes.seek(0)
            # accumulated_audio = np.array([], dtype='int16')  # 清空累积的音频数据，为下一次录音做准备
            # 发送请求到后端
            response = requests.post("http://skaiqwenapi.skaispace.com:5000/transcribe",
                                     files={"audio": ("audio.wav", audio_bytes)})
            if response.ok:
                print("转写结果:", response.json().get('transcription', '转写失败'))
            else:
                print("转写请求失败")
        sleep(0.1)  # 稍微等待，避免CPU过高


# 启动转写线程
threading.Thread(target=transcribe_audio, daemon=True).start()

# 使用sounddevice持续录音
with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=audio_callback):
    print("正在录音，请开始说话...")
    threading.Event().wait()  # 让主线程保持运行
