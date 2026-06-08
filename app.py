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

# Matikan warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Matplotlib tanpa GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

from flask import Flask, request, jsonify, render_template, send_file
import edge_tts

# Load pydub
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

app = Flask(__name__)

# Folder audio sementara
TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), 'tts_audio')
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Hapus file temporary HANYA saat aplikasi pertama kali start
def cleanup_temp_audio_dir_startup():
    for entry in os.listdir(TEMP_AUDIO_DIR):
        path = os.path.join(TEMP_AUDIO_DIR, entry)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

cleanup_temp_audio_dir_startup()

# Label kelas ASR
CLASSES = [
    'Blender', 'Cermin', 'Dispenser', 'Jam', 'Kipas',
    'Kulkas', 'Kursi', 'Lemari', 'Meja', 'Mikrowave',
    'Oven', 'Pintu', 'Sofa', 'Televisi'
]

# Load model ASR
MODEL_PATH = "static/model/model_asr_indonesia_2.h5"
model_asr = None

if os.path.exists(MODEL_PATH):
    try:
        model_asr = tf.keras.models.load_model(MODEL_PATH)
        print("-> [SUKSES] Model ASR berhasil dimuat")
    except Exception as e:
        print(f"-> [ERROR] Gagal load model: {e}")
else:
    print(f"-> [WARNING] File model tidak ditemukan")

# Ekstraksi MFCC
def extract_mfcc(file_path, max_pad_len=44):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Ekstraksi 13 MFCC statis, lalu hitung Delta dan Delta-Delta
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc_features)
        mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)

        # Gabungkan menjadi 39 fitur dinamis
        mfcc = np.vstack((mfcc_features, mfcc_delta, mfcc_delta2))

        # Normalisasi spektral (Mean Subtraction)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

        # Padding atau truncate komponen waktu (frames)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc.astype(np.float32)
    except Exception as e:
        print("Error ekstraksi MFCC:", e)
        return None

# Generate audio TTS
async def generate_audio(text, voice, output_path, output_format, supports_output_format):
    if supports_output_format:
        communicate = edge_tts.Communicate(text, voice, output_format=output_format)
    else:
        communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# Cari ffmpeg lokal / global
def find_ffmpeg():
    candidates = []

    # Cek folder bin lokal
    for name in ('ffmpeg.exe', 'ffmpeg'):
        path = os.path.join(app.root_path, 'bin', name)
        if os.path.isfile(path):
            candidates.append(path)

    # Cek PATH sistem
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        candidates.append(system_ffmpeg)

    # Validasi executable
    for path in candidates:
        try:
            subprocess.run([path, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return path
        except (subprocess.CalledProcessError, OSError):
            continue
    return None

FFMPEG_PATH = find_ffmpeg()

# Konfigurasi pydub
if FFMPEG_PATH and AudioSegment is not None:
    AudioSegment.converter = FFMPEG_PATH
    ffprobe_dir = os.path.dirname(FFMPEG_PATH)

    for name in ('ffprobe.exe', 'ffprobe'):
        fp = os.path.join(ffprobe_dir, name)
        if os.path.isfile(fp):
            AudioSegment.ffprobe = fp
            break

if FFMPEG_PATH:
    print(f"-> [INFO] FFmpeg ditemukan: {FFMPEG_PATH}")
else:
    print("-> [WARNING] FFmpeg tidak ditemukan")

# Konversi audio
def convert_audio_format(source_path, target_path, out_fmt='wav'):
    # Gunakan pydub
    if AudioSegment is not None:
        try:
            audio = AudioSegment.from_file(source_path)
            audio.export(target_path, format=out_fmt)
            return
        except Exception:
            pass

    # Fallback subprocess
    if FFMPEG_PATH:
        subprocess.run([FFMPEG_PATH, '-y', '-i', source_path, target_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    raise RuntimeError('FFmpeg tidak ditemukan')

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint TTS
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

    supports_output_format = ('output_format' in inspect.signature(edge_tts.Communicate).parameters)

    format_map = {
        'mp3': {
            'extension': 'mp3',
            'output_format': 'audio-24khz-48kbitrate-mono-mp3'
        },
        'wav': {
            'extension': 'wav',
            'output_format': 'riff-24khz-16bit-mono-pcm'
        }
    }

    if requested_format not in format_map:
        return jsonify({'error': 'Format audio tidak didukung', 'success': False}), 400

    filename = f"{uuid.uuid4().hex}.{format_map[requested_format]['extension']}"
    output_path = os.path.join(TEMP_AUDIO_DIR, filename)
    temp_mp3_path = None

    if requested_format == 'wav' and not supports_output_format:
        temp_mp3_path = os.path.join(TEMP_AUDIO_DIR, f"{uuid.uuid4().hex}.mp3")

    try:
        # Generate WAV via MP3 sementara
        if requested_format == 'wav' and not supports_output_format:
            asyncio.run(generate_audio(text, voice, temp_mp3_path, format_map['mp3']['output_format'], supports_output_format))
            convert_audio_format(temp_mp3_path, output_path, 'wav')
        else:
            asyncio.run(generate_audio(text, voice, output_path, format_map[requested_format]['output_format'], supports_output_format))

        if not os.path.exists(output_path):
            raise RuntimeError('File audio gagal dibuat')

        with open(output_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()

        mime_map = {'mp3': 'audio/mpeg', 'wav': 'audio/wav'}
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
        # Menghapus secara terisolasi hanya file milik request ini
        if output_path and os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass
        if temp_mp3_path and os.path.exists(temp_mp3_path):
            try: os.remove(temp_mp3_path)
            except OSError: pass

# Endpoint ASR
@app.route('/api/asr', methods=['POST'])
def asr_predict():
    if model_asr is None:
        return jsonify({'error': 'Model ASR belum dimuat', 'success': False}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'File audio tidak ditemukan', 'success': False}), 400

    audio_file = request.files['audio']
    unique_id = uuid.uuid4().hex

    # Ambil extension asli
    ext = os.path.splitext(audio_file.filename)[1]
    if not ext:
        ext = '.webm'

    filename = f"rec_{unique_id}{ext}"
    temp_save_path = os.path.join(TEMP_AUDIO_DIR, filename)
    audio_file.save(temp_save_path)

    # File hasil konversi
    converted_path = os.path.join(TEMP_AUDIO_DIR, f"clean_{unique_id}.wav")

    try:
        convert_audio_format(temp_save_path, converted_path, 'wav')
    except Exception as e:
        print("Error konversi:", e)
        # Proteksi hapus file jika konversi gagal sebelum masuk ke try utama
        if os.path.exists(temp_save_path):
            try: os.remove(temp_save_path)
            except OSError: pass
        return jsonify({'error': f'Konversi audio gagal: {str(e)}', 'success': False}), 500

    try:
        # Ekstraksi MFCC
        mfcc_input = extract_mfcc(converted_path, max_pad_len=44)
        if mfcc_input is None:
            return jsonify({'error': 'Ekstraksi MFCC gagal', 'success': False}), 400

        # Bentuk input model
        mfcc_model_input = mfcc_input[np.newaxis, ..., np.newaxis]

        # Prediksi model
        prediction = model_asr.predict(mfcc_model_input)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = CLASSES[predicted_index]
        confidence = float(prediction[predicted_index] * 100)

        # Top confidence
        top_indices = np.argsort(prediction)[::-1][:5]
        top_confidence = []
        for idx in top_indices:
            top_confidence.append({
                'label': CLASSES[idx],
                'score': float(prediction[idx] * 100)
            })

        # Visualisasi MFCC
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
        return jsonify({'error': f'Gagal menjalankan model: {str(e)}', 'success': False}), 500
    finally:
        # Menghapus secara terisolasi hanya file milik request ini
        if temp_save_path and os.path.exists(temp_save_path):
            try: os.remove(temp_save_path)
            except OSError: pass
        if converted_path and os.path.exists(converted_path):
            try: os.remove(converted_path)
            except OSError: pass

if __name__ == '__main__':
    app.run(debug=True)