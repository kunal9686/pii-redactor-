

# 🔒 PII Redactor API

This project is a **Flask-based web service** that allows users to upload PDF documents and receive **PII-redacted (Personally Identifiable Information)** versions. It uses **OCR**, **transformers-based NER models**, and **PDF manipulation** to ensure sensitive information is identified and redacted from the document accurately.

---

## 📂 Project Structure

```
├── app.py                 # Contains core logic for PII redaction
├── main.py                # Flask server handling file uploads, redaction, and downloads
├── uploads/               # Stores uploaded PDF files
├── processed/             # Stores redacted output PDFs
├── status/                # JSON files tracking redaction status
├── samples/               # Sample PDF files for demo
├── templates/index.html   # Optional frontend (if provided)
├── models/                # Cached HuggingFace models
├── src/                   
│   ├── logger.py          # Custom logging utility
│   └── custom_exception.py# Custom exception class
```

---

## ⚙️ Features

* ✅ Upload a PDF document via an API
* 🧠 Detect PII using a HuggingFace transformer model (`ab-ai/pii_model`)
* 🖍️ Redact detected PII in text and images
* 🧾 Return a redacted PDF for download
* 📈 Track redaction progress via a `/status/<file_id>` endpoint
* 🔍 Preview snippets of original and redacted text

---

## 🚀 Getting Started

### 🔧 Prerequisites

* Python 3.8+
* Tesseract OCR installed and accessible
* Poppler installed for PDF image conversion

**Install requirements:**

```bash
pip install -r requirements.txt
```

**Ensure dependencies are installed and paths are updated:**

* Tesseract Path: Update `pytesseract.pytesseract.tesseract_cmd` in `app.py`
* Poppler Path: Used in `convert_from_path()` — update to your local Poppler bin path

---

### ▶️ Running the App

Start the Flask app:

```bash
python main.py
```

Visit: [http://localhost:5001](http://localhost:5001)

---

## 🧪 API Endpoints

### 📤 `POST /upload`

Upload a PDF for redaction.

* **Form field**: `file` (PDF only)
* **Returns**: `file_id`, `message`

### 📥 `GET /status/<file_id>`

Check redaction progress.

* **Returns**: redaction status, statistics, accuracy, types of detected PII

### 📄 `GET /preview/original/<file_id>`

Get the first 500 characters of the **original** unredacted text.

### 🔒 `GET /preview/redacted/<file_id>`

Get the first 500 characters of the **redacted** text.

### 📦 `GET /download/<file_id>`

Download the redacted PDF once ready.

### 🎯 `GET /sample`

Run redaction on a predefined sample file.

### ✅ `GET /test`

Test if the API is live and responsive.

---

## 🧠 How It Works

1. **Extract text** from uploaded PDF using `pytesseract` with image preprocessing
2. **Detect PII** entities via HuggingFace NER pipeline
3. **Redact text** by replacing PII with block characters (`█`)
4. **Create redacted PDF** using positional OCR data and PIL drawing
5. **Track status** via JSON files and return stats like:

   * Total pages
   * Accuracy of redaction
   * Entity types (e.g., NAME, EMAIL, PHONE)

---

## 📊 Sample Output

```json
{
  "status": "completed",
  "pii_count": 8,
  "accuracy": 0.95,
  "pii_types": {
    "PER": 3,
    "LOC": 3,
    "PHONE": 2
  }
}
```

---

## 🛠 Configuration Tips

* 📍 Update hardcoded file paths like `samples/khushi_bsl.pdf` or Poppler/Tesseract paths for your system.
* 📤 Ensure folders like `/uploads`, `/processed`, `/status` exist (created automatically if missing).

---

## 🧰 Dependencies

* `pytesseract`
* `pdf2image`
* `transformers`
* `torch`
* `flask`
* `flask_cors`
* `reportlab`
* `Pillow`
* `PyPDF2`

---

## 📌 To Do / Enhancements

* Add UI for drag-and-drop PDF uploads
* Dockerize the app for platform-independent deployment
* Add email notifications when redaction is complete
* Use GPU acceleration more efficiently for larger batches

---

## 🧑‍💻 Author

> Built by \[DEVANSH , KUNAL] — focused on privacy-first document handling using deep learning.

