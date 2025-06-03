# RAG-Powered LLM Q&A System with Local Document Integration

Proyek ini adalah sistem cerdas yang menggunakan teknologi Retrieval-Augmented Generation (RAG) dan Large Language Model (LLM) seperti Gemini dari Google. Sistem ini dapat menjawab pertanyaan berdasarkan informasi dari dokumen-dokumen (misalnya PDF) yang Anda simpan secara lokal. Anda bisa berinteraksi dengannya melalui aplikasi web (dibuat dengan Streamlit) atau bot Telegram.

## Kemampuan Utama (Key Features)

- **Intelligent Q&A**: Dapat menjawab pertanyaan Anda berdasarkan isi dokumen yang Anda sediakan.
- **Document Retrieval**: Bisa menemukan dokumen yang relevan dengan permintaan Anda.
- **Bilingual Support**: Mengerti dan merespons dalam bahasa Indonesia dan Inggris.
- **Dual Interface**:
    - **Web Application**: User interface (UI) yang mudah digunakan melalui browser, dibangun dengan Streamlit.
    - **Telegram Bot**: Dapat diakses langsung dari chat Telegram Anda.
- **Efficient Vector Storage**: Menggunakan ChromaDB untuk menyimpan dan mencari informasi (embeddings) dari dokumen Anda dengan cepat.
- **Powered by Gemini API**: Ditenagai oleh model bahasa canggih dari Google (Gemini) untuk jawaban yang baik.
- **Flexible Document Processing**: Mudah memproses (termasuk chunking) dan mengatur dokumen PDF Anda.
- **Conversation Memory**: Menyimpan riwayat obrolan untuk konteks yang lebih baik (di Streamlit dan Telegram Bot).

## Yang Anda Perlukan (Prerequisites)

- Python versi 3.8 atau lebih baru.
- Google Gemini API Key (bisa didapatkan dari Google AI Studio).
- Dokumen-dokumen PDF yang ingin Anda gunakan.
- Telegram Bot Token (jika ingin menggunakan fitur Telegram Bot).

## Cara Instalasi (Installation)

1.  **Clone Repositori atau Buka Folder Proyek**
    Jika Anda baru memulai, clone dulu repositori ini. Jika sudah ada, cukup buka foldernya.

2.  **Buat dan Aktifkan Python Virtual Environment**
    Ini seperti membuat "ruang kerja" khusus untuk proyek ini agar dependencies tidak tercampur.
    -   Buat virtual environment (misalnya, bernama `myenv`):
        ```bash
        python -m venv myenv
        ```
    -   Aktifkan:
        -   Untuk pengguna Windows:
            ```bash
            myenv\Scripts\activate
            ```
        -   Untuk pengguna macOS atau Linux:
            ```bash
            source myenv/bin/activate
            ```

3.  **Instal Dependencies**
    Perintah ini akan menginstal semua library Python yang dibutuhkan oleh proyek dari file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Siapkan API Key dan Token**
    -   Buat file baru bernama `.env` di folder utama proyek.
    -   Isi file tersebut seperti ini, ganti dengan API Key dan Token Anda:
        ```env
        GEMINI_API_KEY=your_gemini_api_key_here
        TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here # Hanya jika pakai Telegram Bot
        ```
    -   Penting: Jangan bagikan isi file `.env` ini ke orang lain atau ke internet (tambahkan ke `.gitignore`).

5.  **Letakkan Dokumen Anda**
    -   Buat folder bernama `documents_retrieval/` di dalam folder proyek (jika belum ada).
    -   Masukkan semua file PDF yang ingin Anda gunakan ke dalam folder `documents_retrieval/` ini.

## Cara Menggunakan (Usage)

### 1. Inisialisasi Sistem & Dokumen (Penting untuk Penggunaan Pertama Kali)
Jalankan script `init_system.py` ini di terminal. Script ini akan memproses dokumen Anda (melakukan chunking dan membuat embeddings) dan menyiapkannya di vector store (ChromaDB).
```bash
python init_system.py
```

### 2. Menjalankan Aplikasi Web (Streamlit App)
Untuk menggunakan user interface web:
```bash
streamlit run app.py
```
-   Setelah itu, buka browser (seperti Chrome atau Firefox) dan pergi ke alamat `http://localhost:8501`.

### 3. Menjalankan Telegram Bot
Untuk menggunakan melalui Telegram:
```bash
python telegram_bot.py
```
-   Pastikan Anda sudah mengisi `TELEGRAM_BOT_TOKEN` di file `.env`.
-   Bot akan aktif dan bisa Anda ajak bicara di Telegram.

### Contoh Pertanyaan (Bisa untuk Web App atau Telegram Bot)

-   "jelaskan tentang [topik dari dokumen Anda]"
-   "kirimkan saya dokumen yang membahas [subjek tertentu]"
-   "What are the main conclusions from [nama dokumen]?"

## Struktur Proyek (Project Structure)

```
RAG/ (Folder Utama Proyek)
├── app.py                 # Kode untuk aplikasi Streamlit (Web UI)
├── telegram_bot.py        # Kode untuk Telegram Bot
├── rag_system.py          # Logika inti sistem RAG & LLM
├── vector_store.py        # Mengatur vector store (ChromaDB)
├── document_processor.py  # Memproses dokumen PDF (chunking, dll.)
├── init_system.py         # Script untuk inisialisasi dan pemrosesan dokumen awal
├── requirements.txt       # Daftar dependencies Python
├── .env                   # Tempat menyimpan API Key dan Token (Rahasia)
├── documents_retrieval/   # Folder tempat Anda menyimpan PDF
├── chroma_db/             # Database untuk vector embeddings (dibuat otomatis oleh ChromaDB)
├── myenv/                 # Folder Python virtual environment Anda
└── README.md              # File panduan ini
```

## Bagaimana Sistem Ini Bekerja (How It Works - Simplified)

1.  **Document Loading & Chunking**: Dokumen PDF dari folder `documents_retrieval/` dibaca, lalu isinya dipecah menjadi potongan-potongan kecil (chunks).
2.  **Embedding Generation**: Setiap chunk teks diubah menjadi representasi numerik (vector embedding) menggunakan model embedding dari Google.
3.  **Vector Storage**: Semua embeddings ini disimpan di vector store (ChromaDB) agar mudah dicari.
4.  **Query Embedding**: Saat Anda bertanya, pertanyaan Anda juga diubah menjadi embedding.
5.  **Similarity Search**: Sistem akan mencari chunks dokumen di vector store yang embeddings-nya paling mirip dengan embedding pertanyaan Anda.
6.  **Response Generation with LLM**: Chunks dokumen yang paling relevan, beserta pertanyaan Anda, dikirim sebagai konteks ke Large Language Model (Gemini). LLM kemudian menghasilkan jawaban atau menentukan dokumen yang diminta.
7.  **Delivering Response**: Jawaban (atau dokumennya) dikirimkan kembali kepada Anda melalui aplikasi Streamlit atau Telegram Bot.

## Pengaturan Lebih Lanjut (Advanced Configuration)

Anda bisa mengubah beberapa parameter jika perlu, misalnya:

-   **Chunking settings** (ukuran chunk, overlap): Di file `document_processor.py`.
-   **LLM parameters** (model Gemini yang dipakai, temperature untuk kreativitas jawaban): Di file `rag_system.py`.
-   **Vector store settings** (nama folder database ChromaDB): Di file `vector_store.py`.

## Kustomisasi (Customization)

### Menambah Dokumen Baru

1.  Taruh PDF baru di folder `documents_retrieval/`.
2.  Jalankan ulang `python init_system.py` agar dokumen baru ikut diproses dan ditambahkan ke vector store.

### Mengubah Prompt untuk LLM

Anda bisa mengubah template atau instruksi awal (prompt) untuk LLM di file `rag_system.py` (dalam metode `_create_prompt_template()`). Ini bisa mengubah gaya bahasa atau fokus jawaban sistem.

### Mengubah Tampilan Web (Streamlit UI)

Tampilan aplikasi web Streamlit bisa diubah dengan mengedit kode CSS di file `app.py`.

## Jika Ada Masalah (Troubleshooting)

-   **API Key Error?**: Pastikan `GEMINI_API_KEY` di file `.env` sudah benar dan aktif.
-   **Telegram Bot Token Error?**: Cek `TELEGRAM_BOT_TOKEN` di `.env` jika pakai Telegram Bot.
-   **Document Loading Issues?**: Pastikan file PDF Anda tidak rusak dan berisi teks (bukan hanya gambar). Dan pastikan ada di folder `documents_retrieval/`.
-   **System Not Finding Info?**: Mungkin dokumen belum diproses atau vector store kosong. Jalankan `python init_system.py`.

### Ingin Mengulang dari Awal (Reset Vector Store)?

1.  Hapus folder `chroma_db/`.
2.  Jalankan lagi `python init_system.py`.

## Keamanan (Security Notes)

-   **JANGAN PERNAH** membagikan API Key atau Token Bot Anda. Simpan baik-baik dan jangan di-commit ke Git.
-   File `.env` sebaiknya selalu ada di dalam `.gitignore`.

## Lisensi (License)

Proyek ini bebas digunakan untuk belajar atau keperluan pribadi. Pastikan Anda punya hak untuk menggunakan dokumen-dokumen yang Anda masukkan ke sistem.

## Bantuan (Support)

Jika ada masalah:
1.  Lihat pesan error yang muncul di terminal saat Anda menjalankan program.
2.  Pastikan semua dependencies dari `requirements.txt` sudah terinstal di virtual environment Anda.
3.  Pastikan Google Gemini API Key Anda masih berlaku dan memiliki kuota.
