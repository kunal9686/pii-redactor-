import os
import uuid
import json
import logging
from threading import Thread
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, send_from_directory
from main import PIIRedactor
# Initialize the redactor
redactor = PIIRedactor()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['STATUS_FOLDER'] = 'status'  # New folder for status tracking
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB limit
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
from flask_cors import CORS
CORS(app, resources={
    r"/upload": {"origins": "*"},
    r"/status/*": {"origins": "*"},
    r"/download/*": {"origins": "*"},
    r"/sample": {"origins": "*"},
    r"/test": {"origins": "*"}
})

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['STATUS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def serve_index():
    return send_file('templates/index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def update_status(file_id, status, data=None):
    """Update processing status for a file"""
    status_file = os.path.join(app.config['STATUS_FOLDER'], f"{file_id}.json")
    status_data = {
        'status': status,
        'timestamp': str(uuid.uuid4()),  # Simple timestamp replacement
        'data': data or {}
    }
    with open(status_file, 'w') as f:
        json.dump(status_data, f)

def get_status(file_id):
    """Get processing status for a file"""
    status_file = os.path.join(app.config['STATUS_FOLDER'], f"{file_id}.json")
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return None

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
        
        # Initialize status
        update_status(file_id, 'processing')
        
        # Process the file in a background thread
        def process_file():
            try:
                # Uncomment the actual processing when ready
                result = redactor.redact_pdf(upload_path, processed_path)
                
                update_status(file_id, 'completed', {
                    'processed_path': processed_path,
                    'original_filename': original_filename,
                    'pii_count': len(result['pii_entities']),
                    'accuracy': result['accuracy'],
                    'page_count': result.get('page_count', 1),
                    'pii_types': result.get('pii_types', {})  # Add this line
                })
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
    status_data = get_status(file_id)
    
    if not status_data:
        return jsonify({'error': 'File not found'}), 404
    
    if status_data['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'processed_path': status_data['data'].get('processed_path'),
            'original_filename': status_data['data'].get('original_filename'),
            'download_filename': f"redacted_{status_data['data'].get('original_filename')}",
            'stats': {
                'pii_count': status_data['data'].get('pii_count', 0),
                'accuracy': status_data['data'].get('accuracy', 0.0),
                'page_count': status_data['data'].get('page_count', 1),
                'pii_types': status_data['data'].get('pii_types', {})  # Add this line
            }
        })
    elif status_data['status'] == 'failed':
        return jsonify({
            'status': 'failed',
            'error': status_data['data'].get('error', 'Unknown error')
        }), 500
    else:
        return jsonify({
            'status': 'processing',
            'step': status_data['data'].get('step', 'processing')
        }), 200

@app.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    status_data = get_status(file_id)
    
    if not status_data or status_data['status'] != 'completed':
        return jsonify({'error': 'File not found or still processing'}), 404
    
    processed_path = status_data['data'].get('processed_path')
    if not processed_path or not os.path.exists(processed_path):
        return jsonify({'error': 'Processed file not found'}), 404
    
    original_filename = status_data['data'].get('original_filename', 'document.pdf')
    download_filename = f"redacted_{original_filename}"
    
    return send_file(
        processed_path,
        as_attachment=True,
        download_name=download_filename,
        mimetype='application/pdf'
    )

@app.route('/preview/original/<file_id>', methods=['GET'])
def get_original_preview(file_id):
    status_data = get_status(file_id)
    if not status_data:
        return jsonify({'error': 'File not found'}), 404
    
    # Get the first 500 characters of the original text
    original_text = status_data['data'].get('original_snippet', '')
    return jsonify({'preview': original_text})

@app.route('/preview/redacted/<file_id>', methods=['GET'])
def get_redacted_preview(file_id):
    status_data = get_status(file_id)
    if not status_data:
        return jsonify({'error': 'File not found'}), 404
    
    # Get the first 500 characters of the redacted text
    redacted_text = status_data['data'].get('redacted_snippet', '')
    return jsonify({'preview': redacted_text})

@app.route('/sample', methods=['GET'])
def process_sample():
    # Check if sample file exists
    sample_path = "samples/khushi_bsl.pdf"
    if not os.path.exists(sample_path):
        return jsonify({'error': 'Sample file not found'}), 404
    
    file_id = "sample_" + str(uuid.uuid4())
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"redacted_{file_id}_sample.pdf")
    
    # Initialize status
    update_status(file_id, 'processing')
    
    # Process the file in a background thread
    def process_sample_file():
        try:
            update_status(file_id, 'processing', {'step': 'extracting_text'})
            
            # Uncomment when PIIRedactor import is fixed
            # result = redactor.redact_pdf(sample_path, processed_path)
            
            # Temporary simulation
            import time
            time.sleep(3)
            
            # Create dummy output file
            with open(processed_path, 'w') as f:
                f.write("Dummy redacted sample PDF content")
            
            update_status(file_id, 'completed', {
                'processed_path': processed_path,
                'original_filename': 'sample.pdf',
                'pii_count': 8,
                'accuracy': 0.95,
                'page_count': 3,
                'pii_types': {  # Add sample PII types
                    'PER': 3,
                    'LOC': 3,
                    'PHONE': 2
                }
            })
            
        except Exception as e:
            logging.error(f"Error processing sample file: {str(e)}")
            update_status(file_id, 'failed', {'error': str(e)})
    
    Thread(target=process_sample_file).start()
    
    return jsonify({
        'file_id': file_id,
        'original_filename': "sample.pdf",
        'message': 'Sample file processing started'
    }), 202

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)