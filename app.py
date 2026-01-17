from flask import Flask, render_template, request, redirect, url_for
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from datetime import datetime
import json

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'database.json')


# --- 1. DATABASE HELPERS ---
def load_db():
    if not os.path.exists(DB_FILE):
        return {"history": []}
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"history": []}


def save_db(data):
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Database error: {e}")


# --- 2. TEXT CLEANING (The Fix) ---
def clean_text(text):
    # Remove special characters and make lowercase
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower()


def get_missing_keywords(resume_text, job_description):
    # This list tells the AI to IGNORE these generic words
    ignore_words = set([
        'and', 'the', 'is', 'in', 'at', 'of', 'a', 'to', 'for', 'with', 'on', 'by', 'an', 'be', 'it',
        'this', 'that', 'are', 'from', 'or', 'as', 'but', 'not', 'can', 'will', 'has', 'have', 'do',
        'we', 'you', 'your', 'my', 'job', 'description', 'role', 'work', 'experience', 'candidate',
        'skills', 'team', 'responsibilities', 'requirements', 'qualifications', 'preferred', 'plus',
        'proficiency', 'proficient', 'ideal', 'title', 'summary', 'years', 'degree', 'bachelor',
        'masters', 'university', 'knowledge', 'strong', 'ability', 'platforms', 'build', 'create',
        'design', 'deploy', 'support', 'maintain', 'ensure', 'looking', 'seeking', 'opportunity',
        'excellent', 'communication', 'track', 'record', 'proven', 'field', 'related', 'computer',
        'science', 'engineering', 'application', 'applications', 'systems', 'solutions', 'tasks'
    ])

    jd_words = set(clean_text(job_description).split())
    resume_words = set(clean_text(resume_text).split())

    # Find words in JD that are NOT in Resume
    missing = jd_words - resume_words

    # Filter out the "ignore words" and short words (less than 3 letters)
    filtered_missing = [word for word in missing if word not in ignore_words and len(word) > 2]

    return list(filtered_missing)[:15]  # Return top 15 missing words


def calculate_match(resume_text, job_description):
    if not resume_text or not job_description:
        return 0.0

    # Clean both texts
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    text = [clean_resume, clean_jd]

    # Use TF-IDF but force it to ignore standard English "stop words"
    cv = TfidfVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text)

    match = cosine_similarity(count_matrix)[0][1] * 100
    return round(match, 2)


# --- 3. ROUTES ---
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
                # Extract text using pdfminer
                resume_text = extract_text(file)

                # Calculate match
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

    return render_template('index.html', match_score=match_score, missing_keywords=missing_keywords, history=history,
                           message=message)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    db = load_db()
    db['history'] = []
    save_db(db)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
