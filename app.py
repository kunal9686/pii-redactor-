import os
import uuid
import logging
from main import PIIRedactor
from threading import Thread
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB limit
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize the redactor (will load the model)
redactor = PIIRedactor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{original_filename}")
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"redacted_{file_id}_{original_filename}")
        
        # Save the file
        file.save(upload_path)
        
        # Process the file in a background thread
        def process_file():
            try:
                result = redactor.redact_pdf(upload_path, processed_path)
                result['processed_path'] = processed_path
                result['original_filename'] = original_filename
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
        
        Thread(target=process_file).start()
        
        return jsonify({
            'file_id': file_id,
            'original_filename': original_filename,
            'message': 'File uploaded and processing started'
        }), 202
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status/<file_id>', methods=['GET'])
def check_status(file_id):
    # In a real app, you'd track processing status in a database
    # For this example, we'll just check if the processed file exists
    processed_files = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if file_id in f]
    
    if processed_files:
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_files[0])
        original_filename = "_".join(processed_files[0].split("_")[2:])
        
        # Get some stats (simplified for example)
        return jsonify({
            'status': 'completed',
            'processed_path': processed_path,
            'original_filename': original_filename,
            'download_filename': f"redacted_{original_filename}",
            'stats': {
                'pii_count': 12,  # In real app, get from processing result
                'accuracy': 0.98,  # In real app, get from processing result
                'page_count': 5    # In real app, get from processing result
            }
        })
    else:
        return jsonify({'status': 'processing'}), 200

@app.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    processed_files = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if file_id in f]
    
    if not processed_files:
        return jsonify({'error': 'File not found or still processing'}), 404
    
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_files[0])
    download_filename = f"redacted_{processed_files[0].split('_', 2)[2]}"
    
    return send_file(
        processed_path,
        as_attachment=True,
        download_name=download_filename,
        mimetype='application/pdf'
    )

@app.route('/sample', methods=['GET'])
def process_sample():
    # Process the sample file
    sample_path = "samples/khushi_bsl.pdf"
    file_id = "sample_" + str(uuid.uuid4())
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"redacted_{file_id}_sample.pdf")
    
    # Process the file in a background thread
    def process_sample_file():
        try:
            result = redactor.redact_pdf(sample_path, processed_path)
            result['processed_path'] = processed_path
            result['original_filename'] = "sample.pdf"
        except Exception as e:
            logging.error(f"Error processing sample file: {str(e)}")
    
    Thread(target=process_sample_file).start()
    
    return jsonify({
        'file_id': file_id,
        'original_filename': "sample.pdf",
        'message': 'Sample file processing started'
    }), 202

if __name__ == '__main__':
    app.run(debug=True)