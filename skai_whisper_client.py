# -*- coding: utf-8 -*-
import math
import os
import struct
import tempfile
from urllib import response

import pyaudio
import wave
import threading
from pynput import keyboard
import requests
import torch
import whisper
# import pyautogui
import argparse
import time


class AudioRecorder:
    def __init__(self, chunk=2048, sample_format=pyaudio.paInt16, channels=1, fs=22050, filename="output.wav", volume_threshold=0.01):
        self.chunk = chunk
        self.sample_format = sample_format
        self.channels = channels
        self.fs = fs
        self.filename = filename
        self.frames = []
        self.recording = False
        self.model = "medium"  # 更改默认模型为 large-v3 medium
        self.language = "zh"  # 设置默认语言为中文
        self.console = "zh"
        self.recording_device = 1  # 根据实际情况修改设备编号
        self.p = pyaudio.PyAudio()
        self.translations = {
            "en": {
                "recording": "🔴 Recording",
                "stopped": "🔄 Stopped recording",
                "tutorial": "to begin or end a recording, press ",
                "halt": "press alt+c to halt the program",
                "transcription": "🗣️",
                "language": "The language is currently set to: ",
                "model": "The model is currently set to: ",
            },
            "zh": {
                "recording": "🔴 录音中",
                "stopped": "🔄 转录中",
                "tutorial": "要开始或结束录音，请按 ",
                "halt": "按 alt+c 停止程序",
                "transcription": "🗣️",
                "language": "当前语言: ",
                "model": "当前模型: ",
            },
            "zh-tw": {
                "recording": "🔴 錄音中",
                "stopped": "🔄 轉錄中",
                "tutorial": "若要開始或結束錄音，請按 ",
                "halt": "按 alt+c 停止程序",
                "transcription": "🗣️",
                "language": "當前語言: ",
                "model": "當前模型: ",
            }
        }
        self.trans = self.translations['zh']
        self.last_sound_time = None  # Time of the last sound detected
        self.volume_threshold = volume_threshold

    def console_language(self, console):
        self.console = console
        self.trans = self.translations[console]
        #print(f"Console language set to: {console}")
        print("Console language set to: {}".format(console))

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            print(self.trans["recording"])
            self.frames = []  # Clear previous recording frames
            threading.Thread(target=self.record).start()
        else:
            print(self.trans["stopped"])

    def calculate_volume(self, data):
        """Calculate the volume of the audio data."""
        # This is a simple calculation and might not be accurate for all types of audio data
        count = len(data)/2
        format = "%dh"%(count)
        shorts = struct.unpack(format, data)
        sum_squares = 0.0
        for sample in shorts:
            n = sample * (1.0/32768)
            sum_squares += n*n
        return math.sqrt(sum_squares / count)

    def record(self, VOLUME_THRESHOLD=None):
        stream = self.p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.fs,
            frames_per_buffer=self.chunk,
            input=True,
        )

        # project_dir = os.path.dirname(os.path.abspath(__file__))  # Get the project directory
        # temp_dir = os.path.join(project_dir, 'audio')
        # os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists
        # temp_dir = tempfile.gettempdir()

        start_time = time.time()
        interval = 0.2  # Set the time interval for sending audio files to the server

        while self.recording:
            data = stream.read(self.fs)  # Read 1 second of audio data

            # # If the length of self.frames is more than 10 seconds, remove the oldest data
            # if len(self.frames) > 10 * self.fs:
            #     self.frames.pop(0)

            volume = self.calculate_volume(data)
            if volume > self.volume_threshold:
                self.frames.append(data)
                self.last_sound_time = time.time()
            # elif self.last_sound_time and time.time() - self.last_sound_time > 2:
            #     self.frames = []

            # If the time interval has passed, save the audio data to a temporary file and send it to the server
            if time.time() - start_time >= interval:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.p.get_sample_size(self.sample_format))
                    wf.setframerate(self.fs)
                    wf.writeframes(b''.join(self.frames))

                # Send the temporary file to the server
                with open(temp_file.name, 'rb') as f:
                    response = requests.post('http://192.168.200.132:5000/transcribe', files={'file': f})
                    try:
                        print(response.json()['transcription'])
                    except requests.exceptions.JSONDecodeError:
                        print("Failed to decode JSON from response")

                # Reset the start time but do not clear the frames
                start_time = time.time()

                # Close and delete the temporary file
                temp_file.close()

                # Delete the temporary file
                os.unlink(temp_file.name)
        stream.stop_stream()
        stream.close()
        self.save_audio()

    def transcribe_recording(self, audio_file=None):
        if audio_file is None:
            audio_file = self.filename
        options = {
            "language": self.language,
            "task": "transcribe"
        }
        result = self.model.transcribe(audio_file, **options)
        if not result["text"]:
            return "無法識別的音訊"
        return result["text"]

    def save_audio(self):
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.sample_format))
            wf.setframerate(self.fs)
            wf.writeframes(b''.join(self.frames))

        transcription = self.transcribe_recording()
        print(self.trans["transcription"], transcription)
        #pyautogui.write(transcription)
        print(transcription)

    def set_hotkey(self, hotkey):
        def on_activate():
            self.toggle_recording()

        def for_canonical(f):
            return lambda k: f(listener.canonical(k))

        hotkey = {keyboard.Key[hotkey]}
        with keyboard.GlobalHotKeys({hotkey: for_canonical(on_activate)}) as listener:
            listener.join()

    def set_language(self, language):
        self.language = language
        print(self.trans["language"], self.language)

    def set_model(self, model_name="medium"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = whisper.load_model(model_name, device=device)
        except Exception as e:
            print(f"Failed to load model '{model_name}' on device '{device}'. Error: {e}")
            return
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        print(f"The model '{model_name}' is set to run on {device}.")

def main():
    parser = argparse.ArgumentParser(
        description='Audio Recorder and Transcriber')
    parser.add_argument('--hotkey', type=str, default='alt+x',
                        help='Hotkey to toggle recording')
    parser.add_argument('--language', type=str, default='zh',
                        help='Language for transcription')
    parser.add_argument('--model', type=str, default='medium',
                        help='Model for transcription')
    parser.add_argument('--console', type=str, default='zh',
                        help='Language showing in console')
    args = parser.parse_args()

    recorder = AudioRecorder()
    recorder.console_language(args.console)
    recorder.set_language(args.language)
    recorder.set_model(args.model)
    recorder.set_hotkey(args.hotkey)


if __name__ == "__main__":
    main()
