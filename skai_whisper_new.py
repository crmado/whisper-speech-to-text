from flask import Flask, request, jsonify
import whisper
import numpy as np
import torch

app = Flask(__name__)

# 加載模型，確保指定到正確的GPU
model = whisper.load_model("medium", device="cuda:2")


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    # 從 POST 請求中讀取音頻文件
    audio_file = request.files['audio']
    audio_data = np.frombuffer(audio_file.read(), dtype=np.int16)

    # 確保音頻數據非空
    if len(audio_data) == 0:
        return jsonify({'error': 'Empty audio data'}), 400

    # 音頻數據需要是 float32 類型，並且值範圍在 -1.0 到 1.0 之間
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

    # 使用 Whisper 模型的 transcribe 方法進行轉寫
    try:
        result = model.transcribe(audio=audio_data)
    except Exception as e:
        app.logger.error(f'Error transcribing audio: {str(e)}')
        return jsonify({'error': 'Failed to transcribe audio'}), 500
    print(result['text'])

    # 返回轉寫結果
    return jsonify({'transcription': result['text']}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
