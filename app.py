from flask import Flask, render_template, request, redirect, url_for
from utils.resume_parser import extract_text_from_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import os
from datetime import datetime

app = Flask(__name__)
DB_FILE = 'database.json'

# --- DATABASE HELPERS ---
def load_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as f:
            json.dump({"history": []}, f)
        return {"history": []}
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"history": []}

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- TEXT PROCESSING ---
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def get_missing_keywords(resume_text, job_description):
    stop_words = set(['and', 'the', 'is', 'in', 'at', 'of', 'a', 'to', 'for', 'with', 'on', 'by', 'an', 'be', 'it', 'this', 'that', 'are', 'from', 'or', 'as', 'but', 'not', 'can', 'will', 'has', 'have', 'job', 'description', 'role', 'work', 'experience', 'candidate', 'skills', 'we', 'you', 'your', 'team', 'responsibilities', 'requirements', 'qualifications'])
    jd_words = set(clean_text(job_description).split())
    resume_words = set(clean_text(resume_text).split())
    missing = jd_words - resume_words
    filtered_missing = [word for word in missing if word not in stop_words and len(word) > 3]
    return list(filtered_missing)[:10]

def calculate_match(resume_text, job_description):
    if not resume_text or not job_description:
        return 0.0
    text = [resume_text, job_description]
    cv = TfidfVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text)
    match = cosine_similarity(count_matrix)[0][1] * 100
    return round(match, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    match_score = None
    message = None
    missing_keywords = []
    
    db = load_db()
    history = db.get('history', [])

    if request.method == 'POST':
        job_description = request.form.get('job_description')
        
        if 'resume' not in request.files:
            return render_template('index.html', message="No file uploaded.", history=history)
            
        file = request.files['resume']
        if file.filename == '':
            return render_template('index.html', message="No selected file.", history=history)

        if file:
            try:
                resume_text = extract_text_from_pdf(file)
                match_score = calculate_match(resume_text, job_description)
                missing_keywords = get_missing_keywords(resume_text, job_description)
                
                # Save to History
                new_record = {
                    "filename": file.filename,
                    "score": match_score,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                history.insert(0, new_record)
                db['history'] = history[:5]
                save_db(db)

            except Exception as e:
                message = f"Error processing file: {str(e)}"

    return render_template('index.html', match_score=match_score, missing_keywords=missing_keywords, history=history, message=message)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    db = load_db()
    db['history'] = []
    save_db(db)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
