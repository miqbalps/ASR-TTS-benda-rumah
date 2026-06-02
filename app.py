import os
import uuid
import asyncio
import inspect
import shutil
import subprocess
import io
import base64
import tempfile
import numpy as np
import librosa

# Matikan log warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Matplotlib headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

from flask import Flask, request, jsonify, render_template, send_file
import edge_tts

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

app = Flask(__name__)

TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), 'tts_audio')
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

def cleanup_temp_audio_dir():
    for entry in os.listdir(TEMP_AUDIO_DIR):
        path = os.path.join(TEMP_AUDIO_DIR, entry)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

cleanup_temp_audio_dir()

# KONFIGURASI & PEMUATAN MODEL ASR
CLASSES = [
    'Blender', 'Cermin', 'Dispenser', 'Jam', 'Kipas', 'Kulkas', 'Kursi', 
    'Lemari', 'Meja', 'Mikrowave', 'Oven', 'Pintu', 'Sofa', 'Televisi'
]

MODEL_PATH = "static/model/model_asr_indonesia.h5"
model_asr = None

if os.path.exists(MODEL_PATH):
    try:
        model_asr = tf.keras.models.load_model(MODEL_PATH)
        print("-> [SUKSES] Model ASR 'model_asr_indonesia.h5' Berhasil Dimuat!")
    except Exception as e:
        print(f"-> [EROR] Gagal memuat file model ASR: {e}")
else:
    print(f"-> [PERINGATAN] File '{MODEL_PATH}' tidak ditemukan di direktori root!")

# Ekstraksi MFCC dengan padding statis
def extract_mfcc(file_path, max_pad_len=44):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            
        return mfcc.astype(np.float32)
    except Exception as e:
        print("Eror saat ekstraksi MFCC:", e)
        return None

# ENGINE ASINKRON TTS
async def generate_audio(text, voice, output_path, output_format, supports_output_format):
    if supports_output_format:
        communicate = edge_tts.Communicate(text, voice, output_format=output_format)
    else:
        communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def convert_audio_format(source_path, target_path, out_fmt='wav'):
    if AudioSegment is not None:
        audio = AudioSegment.from_file(source_path)
        audio.export(target_path, format=out_fmt)
        return

    local_ffmpeg = os.path.join(app.root_path, 'bin', 'ffmpeg.exe')
    ffmpeg_path = local_ffmpeg if os.path.exists(local_ffmpeg) else shutil.which('ffmpeg')

    if ffmpeg_path:
        subprocess.run(
            [ffmpeg_path, '-y', '-i', source_path, target_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return
    raise RuntimeError('Fitur konversi audio membutuhkan pydub+ffmpeg atau ffmpeg terinstal pada sistem.')

# FLASK ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data tidak valid', 'success': False}), 400
        
    text = data.get('text', '').strip()
    voice = data.get('voice', 'id-ID-GadisNeural')
    requested_format = data.get('format', 'mp3').lower()
    
    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong', 'success': False}), 400
        
    supports_output_format = 'output_format' in inspect.signature(edge_tts.Communicate).parameters

    format_map = {
        'mp3': {'extension': 'mp3', 'output_format': 'audio-24khz-48kbitrate-mono-mp3'},
        'wav': {'extension': 'wav', 'output_format': 'riff-24khz-16bit-mono-pcm'}
    }

    if requested_format not in format_map:
        return jsonify({'error': 'Format audio tidak didukung.', 'success': False}), 400

    filename = f"{uuid.uuid4().hex}.{format_map[requested_format]['extension']}"
    output_path = os.path.join(TEMP_AUDIO_DIR, filename)

    temp_mp3_path = None
    if requested_format == 'wav' and not supports_output_format:
        temp_mp3_path = os.path.join(TEMP_AUDIO_DIR, f"{uuid.uuid4().hex}.mp3")
    
    try:
        if requested_format == 'wav' and not supports_output_format:
            asyncio.run(generate_audio(text, voice, temp_mp3_path, format_map['mp3']['output_format'], supports_output_format))
            convert_audio_format(temp_mp3_path, output_path, 'wav')
        else:
            asyncio.run(generate_audio(text, voice, output_path, format_map[requested_format]['output_format'], supports_output_format))

        if not os.path.exists(output_path):
            raise RuntimeError('File audio tidak berhasil dibuat di server.')

        with open(output_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()

        mime_map = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav'
        }
        mime_type = mime_map.get(requested_format, 'application/octet-stream')

        return send_file(
            io.BytesIO(audio_bytes),
            mimetype=mime_type,
            as_attachment=False,
            download_name=f"tts-output.{format_map[requested_format]['extension']}"
        )
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
        if temp_mp3_path and os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
        cleanup_temp_audio_dir()

# ENDPOINT ASR & MFCC
@app.route('/api/asr', methods=['POST'])
def asr_predict():
    if model_asr is None:
        return jsonify({'error': 'Model ASR tidak tersedia/belum dimuat di server.', 'success': False}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'File audio rekaman tidak ditemukan.', 'success': False}), 400

    audio_file = request.files['audio']
    unique_id = uuid.uuid4().hex
    filename = f"rec_{unique_id}.wav"
    temp_save_path = os.path.join(TEMP_AUDIO_DIR, filename)
    
    audio_file.save(temp_save_path)

    # Konversi rekaman agar kompatibel untuk librosa
    converted_path = os.path.join(TEMP_AUDIO_DIR, f"clean_{filename}")
    try:
        convert_audio_format(temp_save_path, converted_path, 'wav')
    except Exception:
        converted_path = temp_save_path

    try:
        # 1. Ekstraksi fitur MFCC untuk prediksi
        mfcc_input = extract_mfcc(converted_path, max_pad_len=44)
        if mfcc_input is None:
            return jsonify({'error': 'Gagal melakukan ekstraksi MFCC dari file audio.', 'success': False}), 400

        # Bentuk ulang ke dimensi model
        mfcc_model_input = mfcc_input[np.newaxis, ..., np.newaxis]

        # Prediksi klasifikasi
        prediction = model_asr.predict(mfcc_model_input)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = CLASSES[predicted_index]
        confidence = float(prediction[predicted_index] * 100)

        # Top-5 kepercayaan
        top_indices = np.argsort(prediction)[::-1][:5]
        top_confidence = []
        for idx in top_indices:
            top_confidence.append({
                'label': CLASSES[idx],
                'score': float(prediction[idx] * 100)
            })

        # 2. Visualisasi MFCC
        audio_raw, sr_raw = librosa.load(converted_path, sr=22050)
        mfcc_visual = librosa.feature.mfcc(y=audio_raw, sr=sr_raw, n_mfcc=40)

        plt.figure(figsize=(6, 3))
        librosa.display.specshow(mfcc_visual, sr=sr_raw, x_axis='time')
        plt.colorbar(format='%+2.0f dB', pad=0.01)
        plt.tight_layout(pad=0)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        plt.close()

        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('ascii')

        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'top_confidence': top_confidence,
            'mfcc_image': img_base64
        })

    except Exception as e:
        return jsonify({'error': f"Gagal mengeksekusi model kecerdasan buatan: {str(e)}", 'success': False}), 500
    finally:
        # Bersihkan file sementara
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)
        if os.path.exists(converted_path) and converted_path != temp_save_path:
            os.remove(converted_path)
        cleanup_temp_audio_dir()

if __name__ == '__main__':
    app.run(debug=True)