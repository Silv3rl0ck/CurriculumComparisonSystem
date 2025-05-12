from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords
import numpy as np
import socket
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── CONFIG ───────────────────────────────────────────────────────────────────
app.config['UPLOAD_FOLDER']     = 'uploads'
app.config['PRELOADED_FOLDER']  = 'preloaded_files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# ── GLOBAL STATE ─────────────────────────────────────────────────────────────
PRELOADED_DATABASE = []
UPLOADED_DATABASE  = []
TFIDF_VECTORIZER   = None
TFIDF_MATRIX       = None
GEMMA_DOC_EMBS     = None

# ── STOPWORDS ─────────────────────────────────────────────────────────────────
STOPWORDS = thai_stopwords() | {'ศึกษา', 'พัฒนา', 'การ', 'เกี่ยวกับ', 'เพื่อ'}

# ── LOAD GEMMA MODEL ─────────────────────────────────────────────────────────
try:
    GEMMA_TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-2b", token=HF_API_TOKEN)
    GEMMA_MODEL = AutoModelForSequenceClassification.from_pretrained(
        "google/gemma-2b",
        token=HF_API_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    print("Gemma loaded.")
except Exception as e:
    print("Gemma load error:", e)
    GEMMA_MODEL = None

# ── PREPROCESS ────────────────────────────────────────────────────────────────
def preprocess_thai_text(text):
    text = normalize(text)
    toks = word_tokenize(text, engine="newmm")
    return ' '.join(w for w in toks if w not in STOPWORDS and len(w) > 2)

# ── TF-IDF & EMBEDDINGS ──────────────────────────────────────────────────────
def initialize_tfidf(docs):
    global TFIDF_VECTORIZER, TFIDF_MATRIX
    proc = [preprocess_thai_text(d) for d in docs]
    TFIDF_VECTORIZER = TfidfVectorizer(ngram_range=(1,3), max_df=0.85, min_df=2, max_features=5000)
    TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(proc)

def rebuild_gemma_embeddings():
    global GEMMA_DOC_EMBS
    if not GEMMA_MODEL or not UPLOADED_DATABASE:
        GEMMA_DOC_EMBS = None
        return
    embs = []
    for c in UPLOADED_DATABASE:
        emb = get_gemma_embedding(c['description'])
        if emb is None:
            emb = np.zeros(GEMMA_MODEL.config.hidden_size)
        embs.append(emb)
    GEMMA_DOC_EMBS = np.vstack(embs)

def rebuild_tfidf_and_embeddings():
    descriptions = [c['description'] for c in UPLOADED_DATABASE]
    initialize_tfidf(descriptions)
    rebuild_gemma_embeddings()

def get_gemma_embedding(text):
    try:
        toks = GEMMA_TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        toks = {k:v.to(GEMMA_MODEL.device) for k,v in toks.items()}
        with torch.no_grad():
            out = GEMMA_MODEL(**toks)
        return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except:
        return None

def calculate_semantic_similarity(query, docs, query_code=None):
    # TF-IDF
    proc = preprocess_thai_text(query)
    q_vec = TFIDF_VECTORIZER.transform([proc])
    tfidf_scores = cosine_similarity(q_vec, TFIDF_MATRIX)[0]

    # Gemma
    if GEMMA_MODEL and GEMMA_DOC_EMBS is not None:
        q_emb = get_gemma_embedding(query)
        if q_emb is not None:
            gemma_scores = cosine_similarity([q_emb], GEMMA_DOC_EMBS)[0]
            α = 0.5
            tfidf_scores = α * tfidf_scores + (1 - α) * gemma_scores

    # Course-code weighting
    codes = [c['id'].split('-')[0] for c in docs]
    weights = {code:(0.7 if query_code and code != query_code else 1.0) for code in set(codes)}
    arr = np.array([weights.get(c, 1.0) for c in codes])
    final_scores = tfidf_scores * arr

    # Top 5
    idxs = final_scores.argsort()[-5:][::-1]
    return [{
        "index":   int(i),
        "score":   float(final_scores[i]),
        "percentage": round(float(final_scores[i]) * 100, 2)
    } for i in idxs]

# ── EXCEL LOADING ────────────────────────────────────────────────────────────
def load_course_database(path, db_type='preloaded'):
    try:
        df = pd.read_excel(path)
        cols = ['รหัสวิชา','ชื่อวิชาภาษาไทย','คำอธิบายรายวิชา']
        if not all(c in df.columns for c in cols):
            return False, "Missing columns"
        courses = []
        for _, r in df.iterrows():
            courses.append({
                'id': str(r['รหัสวิชา']).strip(),
                'name_th': str(r['ชื่อวิชาภาษาไทย']).strip(),
                'description': str(r['คำอธิบายรายวิชา']).strip(),
                'type': db_type
            })
        return True, courses
    except Exception as e:
        return False, str(e)

# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def home():    return render_template('home.html')
@app.route('/textcompare')
def textcompare(): return render_template('textcompare.html')
@app.route('/filecompare')
def filecompare(): return render_template('filecompare.html')
@app.route('/dashboard')
def dashboard():  return render_template('dashboard.html')
@app.route('/preload')
def preload():    return render_template('preload.html')

# ── API: PRELOADED FILES ────────────────────────────────────────────────────
@app.route('/api/preloaded-files')
def api_preloaded_files():
    files = [f for f in os.listdir(app.config['PRELOADED_FOLDER']) if f.lower().endswith('.xlsx')]
    return jsonify(files=files)

@app.route('/api/upload-preloaded', methods=['POST'])
def api_upload_preloaded():
    if 'file' not in request.files:
        return jsonify(success=False, error='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error='No selected file'), 400

    if not file.filename.lower().endswith('.xlsx'):
        return jsonify(success=False, error='Invalid file type'), 400

    # preserve original Thai filename
    filename = os.path.basename(file.filename)
    dest = os.path.join(app.config['PRELOADED_FOLDER'], filename)
    try:
        file.save(dest)
        return jsonify(success=True, filename=filename)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

# ── API: GET COURSE DESCRIPTION ─────────────────────────────────────────────
@app.route('/api/get-course-description', methods=['POST'])
def api_get_course_description():
    data = request.get_json()
    course_name = data.get('courseName', '').strip()
    filename = data.get('fileId', '')
    if not course_name:
        return jsonify(success=False, error='Course name required'), 400
    if not filename:
        return jsonify(success=False, error='No file selected'), 400

    path = os.path.join(app.config['PRELOADED_FOLDER'], filename)
    if not os.path.exists(path):
        return jsonify(success=False, error='File not found'), 404

    try:
        df = pd.read_excel(path)
        row = df[df['ชื่อวิชาภาษาไทย']
                 .astype(str)
                 .str.strip()
                 .str.lower() == course_name.lower()]
        if row.empty:
            return jsonify(success=False, error='Course not found'), 404

        desc = row.iloc[0]['คำอธิบายรายวิชา']
        return jsonify(success=True, courseName=course_name, description=str(desc).strip())
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

# ── API: COMPARE ──────────────────────────────────────────────────────────────
@app.route('/api/compare-with-database', methods=['POST'])
def api_compare_with_database():
    data = request.get_json()
    text = data.get('text', '').strip()
    desc = data.get('description', '').strip()

    if not text and not desc:
        return jsonify(error='Comparison text or description required'), 400
    if not UPLOADED_DATABASE:
        return jsonify(error='No uploaded database loaded'), 400

    query = f"{text} {desc}".strip()
    try:
        sims = calculate_semantic_similarity(query, UPLOADED_DATABASE)
        out = [{
            'id': UPLOADED_DATABASE[r['index']]['id'],
            'name': UPLOADED_DATABASE[r['index']]['name_th'],
            'description': UPLOADED_DATABASE[r['index']]['description'],
            'similarity': r['score'],
            'percentage': r['percentage'],
            'type': UPLOADED_DATABASE[r['index']]['type']
        } for r in sims]
        return jsonify(success=True, results=out, total_courses=len(UPLOADED_DATABASE))
    except Exception as e:
        return jsonify(error=str(e)), 500

# ── API: UPLOAD DB ─────────────────────────────────────────────────────
@app.route('/api/upload-database', methods=['POST'])
def api_upload_database():
    global UPLOADED_DATABASE
    if 'file' not in request.files:
        return jsonify(success=False, error='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False, error='No selected file'), 400
    if not file.filename.lower().endswith('.xlsx'):
        return jsonify(success=False, error='Invalid file type'), 400

    filename = secure_filename(file.filename)
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(tmp_path)
        ok, courses = load_course_database(tmp_path, 'uploaded')
        if not ok:
            return jsonify(success=False, error=courses), 400

        UPLOADED_DATABASE = courses
        rebuild_tfidf_and_embeddings()
        return jsonify(success=True, count=len(courses), message='Upload successful')
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── PRELOAD ON STARTUP ────────────────────────────────────────────────────────
def initialize_preloaded_database():
    os.makedirs(app.config['PRELOADED_FOLDER'], exist_ok=True)
    for fn in os.listdir(app.config['PRELOADED_FOLDER']):
        if fn.lower().endswith('.xlsx'):
            ok, courses = load_course_database(os.path.join(app.config['PRELOADED_FOLDER'], fn))
            if ok:
                PRELOADED_DATABASE.extend(courses)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PRELOADED_FOLDER'], exist_ok=True)
    initialize_preloaded_database()
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"Serving at http://{ip}:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
