from flask import send_from_directory
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
import difflib
import math
import re
import string
from collections import Counter
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize
import numpy as np
from transformers import pipeline
from pythainlp.util import normalize
from collections import defaultdict
import socket
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}
  
# Replace the hardcoded token with the environment variable
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Reduce PyTorch memory footprint
torch.backends.cuda.enable_flash_sdp(True)  
torch.set_float32_matmul_precision('medium')


# Initialize models
try:
    # Initialize the model and tokenizer
    model_id = "google/gemma-2b-it"
    GEMMA_TOKENIZER = AutoTokenizer.from_pretrained(model_id, token=HF_API_TOKEN)
    GEMMA_MODEL = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    token=HF_API_TOKEN,
    torch_dtype=torch.float16,  # 16-bit to save memory
    device_map="auto"
    ).eval()
    
    # Thai-optimized TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        tokenizer=word_tokenize,
        analyzer='word',
        ngram_range=(1, 2)
    )
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    GEMMA_MODEL = None

# Initialize course database and embedding cache
COURSE_DATABASE = []
EMBEDDING_CACHE = defaultdict(dict)

def get_gemma_embedding(text, cache_key=None):
    """Get embedding from Gemma model with caching"""
    if cache_key and cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]
    
    try:
        inputs = GEMMA_TOKENIZER(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(GEMMA_MODEL.device)
        
        with torch.no_grad():
            outputs = GEMMA_MODEL(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        if cache_key:
            EMBEDDING_CACHE[cache_key] = embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def enhanced_thai_similarity(text1, text2):
    """Specialized similarity calculation for Thai text"""
    # Normalize Thai characters first
    text1 = normalize(text1)
    text2 = normalize(text2)
    
    # 1. Check for near-identical matches
    seq = difflib.SequenceMatcher(None, text1, text2)
    if seq.ratio() >= 0.95:
        return 1.0
    
    # 2. Thai-specific preprocessing
    def preprocess_thai(text):
        text = re.sub(r'[^\u0E00-\u0E7F\s]', '', text)
        words = word_tokenize(text, engine="newmm")
        return ' '.join(words)
    
    tokenized1 = preprocess_thai(text1)
    tokenized2 = preprocess_thai(text2)
    
    # 3. Combined similarity metrics
    set1 = set(tokenized1.split())
    set2 = set(tokenized2.split())
    jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
    
    seq_ratio = seq.ratio()
    
    tfidf_matrix = tfidf.fit_transform([tokenized1, tokenized2])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return (0.3 * seq_ratio) + (0.3 * jaccard) + (0.4 * tfidf_score)



# Helper functions for text similarity
def tokenize(text):
    """Convert text to lowercase and tokenize into words"""
    return re.findall(r'\w+', text.lower()) if text else []

def calculate_jaccard(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    set1 = set(tokenize(text1))
    set2 = set(tokenize(text2))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def calculate_cosine(text1, text2):
    """Calculate Cosine similarity between two texts"""
    vec1 = Counter(tokenize(text1))
    vec2 = Counter(tokenize(text2))
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    return numerator / denominator if denominator else 0

def calculate_levenshtein(text1, text2):
    """Calculate Levenshtein similarity between two texts"""
    m = len(text1)
    n = len(text2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
        
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            cost = 0 if text1[i-1] == text2[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1,      # deletion
                          d[i][j-1] + 1,      # insertion
                          d[i-1][j-1] + cost) # substitution
    
    distance = d[m][n]
    max_len = max(m, n)
    return 1 - (distance / max_len) if max_len else 0

def calculate_semantic_similarity(text1, text2):
    """Enhanced hybrid similarity calculation"""
    # First check for exact matches
    if text1.strip() == text2.strip():
        return 1.0
    
    # Get Gemma embeddings
    emb1 = get_gemma_embedding(text1, f"input_{hash(text1)}")
    emb2 = get_gemma_embedding(text2, f"course_{hash(text2)}")
    
    if emb1 is not None and emb2 is not None:
        gemma_sim = cosine_similarity([emb1], [emb2])[0][0]
        # If Gemma is very confident, return its result
        if gemma_sim > 0.35 or gemma_sim < 0.15:
            return gemma_sim
    
    # Fall back to Thai-specific methods
    thai_sim = enhanced_thai_similarity(text1, text2)
    
    # Blend results if we have both
    if emb1 is not None and emb2 is not None:
        return (gemma_sim * 0.6) + (thai_sim * 0.4)
    
    return thai_sim

# Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/textcompare')
def text_comparison():
    """Render the text comparison page"""
    return render_template('textcompare.html')

@app.route('/filecompare')
def file_comparison():
    """Render the file comparison page"""
    return render_template('filecompare.html')



@app.route('/api/compare-text', methods=['POST'])
def api_compare_text():
    data = request.get_json()
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    algorithm = data.get('algorithm', 'jaccard')
    
    if not text1 or not text2:
        return jsonify({'error': 'Both text fields are required'}), 400
    
    if algorithm == 'gemma':
        score = calculate_semantic_similarity(text1, text2)
    elif algorithm == 'jaccard':
        score = calculate_jaccard(text1, text2)
    elif algorithm == 'cosine':
        score = calculate_cosine(text1, text2)
    elif algorithm == 'levenshtein':
        score = calculate_levenshtein(text1, text2)
    else:
        return jsonify({'error': 'Invalid algorithm specified'}), 400
    
    return jsonify({
        'similarity': score,
        'percentage': round(score * 100, 2)
    })

@app.route('/api/get-course-description', methods=['POST'])
def get_course_description():
    data = request.get_json()
    course_name = data.get('courseName', '').strip()
    file_id = data.get('fileId', '')
    
    if not course_name:
        return jsonify({'success': False, 'error': 'Course name is required'}), 400
    
    if not file_id:
        return jsonify({'success': False, 'error': 'File selection is required'}), 400
    
    try:
        # Map file IDs to actual filenames
        file_mapping = {
            '1': '1.ฝั่งที่ต้องการเทียบ-ปวส-มทรส-แบบไม่ตัดคำ.xlsx'
        }
        
        filename = file_mapping.get(file_id)
        if not filename:
            return jsonify({'success': False, 'error': 'Invalid file selection'}), 400
        
        file_path = os.path.join('preloaded_files', filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        df = pd.read_excel(file_path)
        
        # Find course by name (case insensitive)
        course = df[df['ชื่อวิชาภาษาไทย'].str.strip().str.lower() == course_name.lower()]
        
        if course.empty:
            return jsonify({'success': False, 'error': 'Course not found'}), 404
        
        description = course.iloc[0]['คำอธิบายรายวิชา']
        return jsonify({
            'success': True,
            'description': str(description).strip(),
            'courseName': course_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Update the compare-with-database endpoint
@app.route('/api/compare-with-database', methods=['POST'])
def api_compare_with_database():
    data = request.get_json()
    comparison_text = data.get('text', '').strip()
    
    if not comparison_text:
        return jsonify({'error': 'Comparison text is required'}), 400
    
    if not COURSE_DATABASE:
        return jsonify({'error': 'No course database loaded'}), 400
    
    try:
        results = []
        batch_size = 8
        
        # Get input embedding once
        input_embedding = get_gemma_embedding(comparison_text, f"input_{hash(comparison_text)}")
        
        # Process in batches
        for i in range(0, len(COURSE_DATABASE), batch_size):
            batch = COURSE_DATABASE[i:i + batch_size]
            
            for course in batch:
                combined_course_text = f"{course['name_th']} {course['description']}"
                cache_key = f"course_{course['id']}"
                
                if input_embedding is not None:
                    course_embedding = get_gemma_embedding(combined_course_text, cache_key)
                    if course_embedding is not None:
                        similarity = cosine_similarity([input_embedding], [course_embedding])[0][0]
                    else:
                        similarity = enhanced_thai_similarity(comparison_text, combined_course_text)
                else:
                    similarity = enhanced_thai_similarity(comparison_text, combined_course_text)
                
                results.append({
                    'id': course['id'],
                    'name': course['name_th'],
                    'description': course['description'],
                    'similarity': similarity,
                    'percentage': round(similarity * 100, 2)
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        for i, res in enumerate(results[:5]):
            res['rank'] = i + 1
        
        return jsonify({
            'success': True,
            'results': results[:5],
            'total_courses': len(COURSE_DATABASE)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def load_preloaded_file(filename):
    """Load a preloaded Excel file from the preloaded_files directory"""
    try:
        # Map the dropdown selection to actual filenames
        file_mapping = {
            '1': '1.ฝั่งที่ต้องการเทียบ-ปวส-มทรส-แบบไม่ตัดคำ.xlsx'
        }
        
        actual_filename = file_mapping.get(filename)
        if not actual_filename:
            return []
            
        file_path = os.path.join('preloaded_files', actual_filename)
        if not os.path.exists(file_path):
            return []
            
        df = pd.read_excel(file_path)
        
        # Verify required columns
        required_columns = ['รหัสวิชา', 'ชื่อวิชาภาษาไทย', 'คำอธิบายรายวิชา']
        if not all(col in df.columns for col in required_columns):
            return []
            
        # Process data
        courses = []
        for _, row in df.iterrows():
            courses.append({
                'id': str(row['รหัสวิชา']).strip(),
                'name_th': str(row['ชื่อวิชาภาษาไทย']).strip(),
                'description': str(row['คำอธิบายรายวิชา']).strip(),
                'category': 'Preloaded Database'
            })
            
        return courses
        
    except Exception as e:
        print(f"Error loading preloaded file: {str(e)}")
        return []


# In app.py, before the upload route
@app.route('/api/check-upload', methods=['GET'])
def check_upload():
    return jsonify({
        'status': 'OK',
        'upload_folder': os.path.abspath(app.config['UPLOAD_FOLDER']),
        'writeable': os.access(app.config['UPLOAD_FOLDER'], os.W_OK)
    })

@app.route('/api/upload-database', methods=['POST'])
def upload_database():
    global COURSE_DATABASE
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
        
    file = request.files['file']
    
    # If user does not select file
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.xlsx'):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(temp_path)
            df = pd.read_excel(temp_path)
            
            # Verify required columns
            required_columns = ['รหัสวิชา', 'ชื่อวิชาภาษาไทย', 'คำอธิบายรายวิชา']
            if not all(col in df.columns for col in required_columns):
                return jsonify({
                    'success': False,
                    'error': 'Missing required columns'
                }), 400
                
            # Process data
            courses = []
            for _, row in df.iterrows():
                courses.append({
                    'id': str(row['รหัสวิชา']).strip(),
                    'name_th': str(row['ชื่อวิชาภาษาไทย']).strip(),
                    'description': str(row['คำอธิบายรายวิชา']).strip(),
                    'category': 'Uploaded Database'
                })
            
            COURSE_DATABASE = courses
            return jsonify({
                'success': True,
                'count': len(courses),
                'message': 'Upload successful'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    return jsonify({'success': False, 'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Get local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"Starting server... accessible at http://{local_ip}:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)


