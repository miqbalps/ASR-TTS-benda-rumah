<p align="center">
  <img src="static/images/logo-tts.png" alt="ListenSpeech Logo" width="80">
</p>

<h1 align="center">ListenSpeech</h1>

<p align="center">
  <strong>Aplikasi Text-to-Speech & Automatic Speech Recognition Bahasa Indonesia</strong><br>
  Proyek Tugas Akhir Kelompok 6 — Pengantar Teknologi Utama (PTU) 2026
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## 📋 Deskripsi

**ListenSpeech** adalah aplikasi web berbasis Flask yang mengintegrasikan dua fitur utama pemrosesan suara:

1. **Text-to-Speech (TTS)** — Mengonversi teks Bahasa Indonesia menjadi audio sintetis menggunakan teknologi Microsoft Edge TTS, dengan opsi pemilihan gender suara, kecepatan bicara, dan format unduhan (MP3/WAV).

2. **Automatic Speech Recognition (ASR)** — Mengenali ucapan pengguna melalui mikrofon dan mengklasifikasikannya ke dalam 14 kategori benda rumah tangga menggunakan model deep learning berbasis MFCC (Mel-Frequency Cepstral Coefficients).

Setelah prediksi ASR berhasil, sistem secara otomatis mengisi deskripsi benda ke kolom TTS dan memainkan audio deskripsinya.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🗣️ **Text-to-Speech** | Konversi teks Bahasa Indonesia ke audio dengan suara natural (Edge TTS) |
| 🎙️ **Speech Recognition** | Rekam suara dari mikrofon, prediksi kelas benda rumah tangga |
| 📊 **Visualisasi MFCC** | Tampilan grafik spektrogram MFCC dari audio yang direkam |
| 🔄 **Auto-Play TTS** | Deskripsi benda otomatis dibacakan setelah prediksi berhasil |
| 🎛️ **Pengaturan Suara** | Pilihan gender (Perempuan/Laki-laki), kecepatan (0.25x–2x) |
| 📥 **Export Audio** | Unduh hasil konversi dalam format MP3 atau WAV |
| 📈 **Top-5 Confidence** | Menampilkan 5 prediksi teratas beserta skor kepercayaan |

---

## 🏗️ Arsitektur & Teknologi

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (Browser)                │
│  HTML + TailwindCSS + Vanilla JavaScript            │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │   Text-to-Speech │  │   Speech Recognition     │ │
│  │   Input & Player │  │   Recorder & Visualizer  │ │
│  └────────┬─────────┘  └────────────┬─────────────┘ │
└───────────┼─────────────────────────┼───────────────┘
            │ /api/synthesize         │ /api/asr
            ▼                         ▼
┌─────────────────────────────────────────────────────┐
│                  Backend (Flask)                     │
│  ┌──────────────┐         ┌───────────────────────┐ │
│  │  Edge TTS     │         │  TensorFlow Model     │ │
│  │  (Microsoft)  │         │  + Librosa MFCC       │ │
│  └──────────────┘         └───────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

| Layer | Teknologi |
|---|---|
| **Frontend** | HTML5, TailwindCSS CDN, JavaScript (Vanilla) |
| **Backend** | Python 3.10+, Flask |
| **TTS Engine** | `edge-tts` (Microsoft Edge Neural Voices) |
| **ASR Model** | TensorFlow/Keras (CNN), MFCC feature extraction via `librosa` |
| **Audio Processing** | `pydub`, `ffmpeg`, `librosa` |
| **Visualisasi** | `matplotlib`, `librosa.display` |

---

## 📁 Struktur Proyek

```
TTS/
├── app.py                          # Aplikasi utama Flask (routes & logic)
├── requirements.txt                # Daftar dependensi Python
├── README.md                       # Dokumentasi proyek
│
├── bin/
│   ├── ffmpeg / ffmpeg.exe         # Binary ffmpeg untuk konversi audio
│   └── ffprobe / ffprobe.exe       # Binary ffprobe (dibutuhkan pydub)
│
├── static/
│   ├── images/
│   │   └── logo-tts.png            # Logo aplikasi
│   └── model/
│       └── model_asr_indonesia.h5  # Model deep learning ASR (~7 MB)
│
└── templates/
    └── index.html                  # Halaman utama (UI)
```

---

## 🚀 Instalasi & Menjalankan

### Prasyarat

- **Python** 3.10 atau lebih baru
- **pip** (Python package manager)
- **ffmpeg** terinstal di sistem atau tersedia di folder `bin/`

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/miqbalps/ASR-TTS-benda-rumah.git
cd ASR-TTS-benda-rumah

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependensi
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
# Jalankan server Flask
flask --app app run
```

Aplikasi akan berjalan di `http://127.0.0.1:5000`.

> [!NOTE]
> Pastikan file model `model_asr_indonesia.h5` berada di `static/model/` agar fitur Speech Recognition berfungsi.

---

## 🔌 API Endpoints

### `GET /`

Menampilkan halaman utama aplikasi.

---

### `POST /api/synthesize`

Mengonversi teks menjadi audio.

**Request Body** (JSON):

```json
{
  "text": "Halo, selamat datang",
  "voice": "id-ID-GadisNeural",
  "format": "mp3"
}
```

| Parameter | Tipe | Deskripsi |
|---|---|---|
| `text` | `string` | Teks yang akan dikonversi (maks. 4500 karakter) |
| `voice` | `string` | Voice ID: `id-ID-GadisNeural` (perempuan) atau `id-ID-ArdiNeural` (laki-laki) |
| `format` | `string` | Format output: `mp3` atau `wav` |

**Response**: File audio binary (`audio/mpeg` atau `audio/wav`).

---

### `POST /api/asr`

Menerima rekaman audio dan mengembalikan prediksi klasifikasi.

**Request Body** (FormData):

| Field | Tipe | Deskripsi |
|---|---|---|
| `audio` | `file` | File audio rekaman (format WAV) |

**Response** (JSON):

```json
{
  "success": true,
  "predicted_label": "Kursi",
  "confidence": 95.42,
  "top_confidence": [
    { "label": "Kursi", "score": 95.42 },
    { "label": "Meja", "score": 2.15 },
    { "label": "Sofa", "score": 1.30 },
    { "label": "Lemari", "score": 0.68 },
    { "label": "Televisi", "score": 0.25 }
  ],
  "mfcc_image": "<base64-encoded-png>"
}
```

---

## 🏷️ Kelas Prediksi ASR

Sistem mengenali **14 kelas benda rumah tangga** berikut:

| No | Kelas | Deskripsi |
|:---:|---|---|
| 1 | **Blender** | Alat elektronik dapur untuk menghaluskan bahan makanan dan minuman |
| 2 | **Cermin** | Benda reflektif untuk memantulkan bayangan |
| 3 | **Dispenser** | Alat pengeluaran air minum panas/dingin dari galon |
| 4 | **Jam** | Alat penunjuk waktu (dinding, tangan, meja) |
| 5 | **Kipas** | Alat penghasil aliran udara penyejuk |
| 6 | **Kulkas** | Lemari es untuk menyimpan dan mengawetkan makanan |
| 7 | **Kursi** | Perabot tempat duduk dengan sandaran |
| 8 | **Lemari** | Perabot penyimpanan pakaian dan barang |
| 9 | **Meja** | Perabot permukaan datar untuk alas kerja |
| 10 | **Mikrowave** | Alat dapur pemanas makanan dengan gelombang mikro |
| 11 | **Oven** | Alat pemanggang tertutup untuk memasak makanan |
| 12 | **Pintu** | Konstruksi akses keluar masuk ruangan |
| 13 | **Sofa** | Tempat duduk empuk untuk bersantai |
| 14 | **Televisi** | Perangkat elektronik penampil siaran gambar dan suara |

---

## 🔧 Konfigurasi Tambahan

### Pengaturan FFmpeg

Konversi format audio (rekaman mikrofon browser → WAV) **membutuhkan `ffmpeg`**. Saat startup, aplikasi mencari dan **memvalidasi** binary secara berurutan:

1. `bin/ffmpeg.exe` atau `bin/ffmpeg` (bundled di proyek)
2. `ffmpeg` dari system PATH

Binary yang ditemukan akan ditest dengan `ffmpeg -version` — jika gagal (misalnya binary Linux di Windows), otomatis diskip ke kandidat berikutnya.

> [!IMPORTANT]
> **Jika fitur Speech Recognition error**, pastikan:
> - File `ffmpeg.exe` dan `ffprobe.exe` ada di folder `bin/` (download dari [ffmpeg.org](https://www.gyan.dev/ffmpeg/builds/))
> - Atau install ffmpeg secara global dan tambahkan ke system PATH
>
> Tanpa `ffmpeg`, rekaman audio dari browser tidak dapat dikonversi ke format yang bisa dibaca oleh model ASR.

### Pengaturan Model ASR

Model harus ditempatkan di:

```
static/model/model_asr_indonesia.h5
```

Model menggunakan arsitektur CNN dengan input MFCC berdimensi `(40, 44, 1)`.

---

## 👥 Tim Pengembang

**Kelompok 6 — Pengantar Teknologi Utama (PTU) 2026**

| NPM | Nama |
|---|---|
| 152023136 | Hickham Amwala Koswara |
| 152023174 | Muhammad Iqbal Pasha Al Farabi |
| 152023214 | Delisya Pramesti Fitriya |

---

## 📄 Lisensi

Hak cipta © 2026 Kelompok 6 PTU. All rights reserved.
