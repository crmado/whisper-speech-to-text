import os
import tempfile

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import skai_whisper_client  # Import the skai_whisper_client module

app = Flask(__name__)
recorder = skai_whisper_client.AudioRecorder()  # Use skai_whisper_client.AudioRecorder to create the recorder instance
recorder.set_model("medium")  # Set the model right after creating the recorder instance


@app.route('/start_recording', methods=['POST'])
def start_recording():
    recorder.toggle_recording()
    return jsonify(message="Recording started"), 200


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    recorder.toggle_recording()
    transcription = recorder.transcribe_recording()
    return jsonify(transcription=transcription), 200


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    if file:
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        transcription = recorder.transcribe_recording(file_path)
        return jsonify(transcription=transcription), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
