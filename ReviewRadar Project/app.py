from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from langdetect import detect
from googletrans import Translator
from flask_sqlalchemy import SQLAlchemy
import os
import json

# Download required NLTK data (quiet)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
translator = Translator()
LEXICON = {}

# DB model
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    translated_text = db.Column(db.Text, nullable=False)
    detected_lang = db.Column(db.String(10), nullable=False)
    overall_sentiment = db.Column(db.String(16), nullable=False)
    aspect_sentiments = db.Column(db.Text, nullable=True)  # store as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

with app.app_context():
    db.create_all()

def load_marathi_lexicon():
    """Load curated Marathi→English lexicon from lexicons/marathi_lexicon.json if present."""
    global LEXICON
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, 'lexicons', 'marathi_lexicon.json'),
            os.path.join(base_dir, 'marathi_lexicon.json')
        ]
        for path in candidates:
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        LEXICON = {str(k).lower(): str(v) for k, v in data.items()}
                        break
    except Exception:
        LEXICON = {}

load_marathi_lexicon()

# Load the trained model and TF-IDF vectorizer
# Ensure 'model/model.pkl' exists and contains (model, tfidf_vectorizer)
MODEL_PATH = os.path.join('model', 'model.pkl')
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place model.pkl with (model, tfidf_vectorizer).")

with open(MODEL_PATH, 'rb') as f:
    model, tfidf_vectorizer = pickle.load(f)

def clean_text(text):
    """Clean text by removing special characters and converting to lowercase"""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))
    text = text.lower()
    return text

def normalize_text(text):
    """Normalize text using lemmatization"""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    normalized_words = [lemmatizer.lemmatize(word) for word in words]
    normalized_text = " ".join(normalized_words)
    return normalized_text

def predict_sentiment(text):
    """Predict sentiment using the same preprocessing as training"""
    text_vectorized = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return prediction

def normalize_hinglish(text):
    """Map common Hinglish and romanized Marathi sentiment words to English equivalents."""
    mappings = {
        # Hindi/Hinglish
        'accha': 'good', 'acha': 'good', 'achha': 'good', 'aacha': 'good',
        'bura': 'bad', 'buraa': 'bad', 'buri': 'bad', 'kharaab': 'bad', 'kharab': 'bad',
        'bekar': 'bad', 'bekaar': 'bad', 'ghatiya': 'bad', 'ghatiyaa': 'bad',
        'mast': 'great', 'jabardast': 'excellent', 'jabardastt': 'excellent',
        'sahi': 'good', 'badhiya': 'good', 'badiya': 'good', 'bakwas': 'terrible',
        'behtar': 'better', 'behterin': 'excellent',
        # Marathi (romanized)
        'chan': 'good', 'chaan': 'good', 'chhan': 'good', 'chhaan': 'good',
        'changle': 'good', 'changla': 'good', 'changlya': 'good', 'utkam': 'excellent',
        'uttam': 'excellent', 'sarvottam': 'excellent', 'sundar': 'beautiful',
        'khup': 'very', 'khupach': 'very', 'far': 'very', 'ekdam': 'very',
        'vait': 'bad', 'wajt': 'bad', 'vaait': 'bad', 'kharab': 'bad', 'nikrushta': 'poor',
        'nikammi': 'useless', 'bekaar': 'bad'
    }
    merged = dict(mappings)
    for k, v in LEXICON.items():
        merged[k] = v

    tokens = word_tokenize(text)
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        low = t.lower()
        repl = None
        if i + 1 < len(tokens):
            bigram = f"{low} {tokens[i+1].lower()}"
            if bigram in merged:
                repl = merged[bigram]
                out.append(repl)
                i += 2
                continue
        repl = merged.get(low, t)
        out.append(repl)
        i += 1
    return " ".join(out)

def detect_and_translate(text):
    """Detect language and translate to English if needed."""
    try:
        lang = detect(text)
    except Exception:
        lang = 'en'

    translated = text
    if lang != 'en':
        try:
            translated = translator.translate(text, dest='en').text
            if (translated or '').strip().lower() == (text or '').strip().lower():
                lang = 'en'
        except Exception:
            translated = text
            lang = 'en'
    return lang, translated

def extract_aspects(text, max_aspects=10):
    """Extract noun and short noun-phrase aspects using POS tagging."""
    lemmatizer = WordNetLemmatizer()
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
    except Exception:
        tagged = []

    chunks = []
    current = []
    for token, tag in tagged:
        if tag.startswith('NN'):
            lemma = lemmatizer.lemmatize(token.lower())
            if lemma.isalpha():
                current.append(lemma)
                if len(current) == 2:
                    chunks.append(" ".join(current))
                    current = []
            else:
                if current:
                    chunks.append(" ".join(current))
                    current = []
        else:
            if current:
                chunks.append(" ".join(current))
                current = []
    if current:
        chunks.append(" ".join(current))

    singles = []
    for token, tag in tagged:
        if tag.startswith('NN'):
            lemma = lemmatizer.lemmatize(token.lower())
            if lemma.isalpha():
                singles.append(lemma)

    phrases_set = set(chunks)
    aspects_ordered = []
    seen = set()
    for p in chunks:
        if p not in seen:
            seen.add(p)
            aspects_ordered.append(p)
    for s in singles:
        is_suffix = any(p.endswith(" " + s) or p == s for p in phrases_set)
        if not is_suffix and s not in seen:
            seen.add(s)
            aspects_ordered.append(s)

    fallback_aspects = [
        'product','quality','price','delivery','color','colour','size','battery','camera',
        'service','design','display','performance','software','sound','material','fit',
        'comfort','shipping','packaging','warranty'
    ]
    text_lower = " ".join([t for t, _ in tagged]).lower()
    for fa in fallback_aspects:
        if fa in text_lower and fa not in seen:
            seen.add(fa)
            aspects_ordered.append(fa)

    return aspects_ordered[:max_aspects]

def _split_clauses(sentence):
    parts = re.split(r"\bbut\b|\bhowever\b|\bthough\b|\balthough\b", sentence, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

def _window_around(tokenized, target_words, radius=5):
    indices = []
    for i, tok in enumerate(tokenized):
        for tw in target_words:
            tw_tokens = tw.split()
            if tw == tok or (" " in tw and tw == " ".join(tokenized[i:i+len(tw_tokens)])):
                indices.append((i, len(tw_tokens)))
                break
    if not indices:
        return " "
    i, span = indices[0]
    start = max(0, i - radius)
    end = min(len(tokenized), i + span + radius)
    return " ".join(tokenized[start:end])

def analyze_aspects(text):
    """
    Return dict of aspect -> sentiment using VADER on local clause/window.
    """
    sia = SentimentIntensityAnalyzer()
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [text]

    aspects = extract_aspects(text)
    aspect_scores = {a: [] for a in aspects}

    for sent in sentences:
        sent_lower = sent.lower()
        clauses = _split_clauses(sent)
        tokenized = word_tokenize(sent_lower)
        for a in aspects:
            a_lower = a.lower()
            clause_for_aspect = None
            for c in clauses:
                if a_lower in c.lower():
                    clause_for_aspect = c
                    break
            if clause_for_aspect:
                score = sia.polarity_scores(clause_for_aspect)["compound"]
                aspect_scores[a].append(score)
            else:
                window = _window_around(tokenized, [a_lower], radius=5)
                if window.strip():
                    score = sia.polarity_scores(window)["compound"]
                    aspect_scores[a].append(score)

    aspect_sentiment = {}
    for a, scores in aspect_scores.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        if avg > 0.05:
            label = 'Positive'
        elif avg < -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        aspect_sentiment[a] = label

    # Normalize aspect keys
    normalized_display = {}
    for a, label in aspect_sentiment.items():
        display_key = normalize_hinglish(a)
        normalized_display[display_key] = label

    # Remove duplicates and overlapping aspects (prefer shorter or clearer term)
    final_aspects = {}
    keys = list(normalized_display.keys())
    for key in keys:
        k = key.strip().lower()
        # If this key is contained as part of a longer key, skip it (we prefer longer phrase)
        is_subsumed = any((k != other and (k in other.split() or k in other)) for other in keys)
        # If both "battery" and "battery life" exist, prefer "battery life" (longer)
        if is_subsumed:
            continue
        if k not in final_aspects:
            final_aspects[k] = normalized_display[key]

    # If final_aspects ended up empty (rare), fall back to normalized_display
    if not final_aspects:
        final_aspects = {k.strip().lower(): v for k, v in normalized_display.items()}

    return final_aspects

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            return render_template('result.html', text='', sentiment='', aspect_sentiments={})
        normalized = normalize_hinglish(text)
        detected_lang, translated = detect_and_translate(normalized)
        translated_norm = normalize_hinglish(translated)
        sentiment = predict_sentiment(translated_norm)
        aspect_sentiments = analyze_aspects(translated_norm)
        return render_template(
            'result.html',
            text=text,
            sentiment=sentiment,
            aspect_sentiments=aspect_sentiments,
            detected_lang=detected_lang,
            translated_text=None if detected_lang == 'en' else translated_norm
        )

@app.route('/absa', methods=['POST'])
def absa():
    payload = request.get_json(silent=True) or {}
    text = payload.get('text', '')
    if not text:
        return jsonify({"error": "text is required"}), 400
    normalized = normalize_hinglish(text)
    detected_lang, translated = detect_and_translate(normalized)
    translated_norm = normalize_hinglish(translated)
    aspects = analyze_aspects(translated_norm)
    overall = predict_sentiment(translated_norm)
    return jsonify({
        "text": text,
        "detected_lang": detected_lang,
        "translated_text": translated_norm if detected_lang != 'en' else None,
        "overall_sentiment": overall,
        "aspects": aspects
    })

@app.route('/submit_review', methods=['POST'])
def submit_review():
    payload = request.get_json(silent=True) or {}
    text = payload.get('text', '').strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    detected_lang, translated = detect_and_translate(text)
    overall = predict_sentiment(translated)
    aspects = analyze_aspects(translated)

    try:
        rec = Review(
            original_text=text,
            translated_text=translated,
            detected_lang=detected_lang,
            overall_sentiment=overall,
            aspect_sentiments=str(aspects)
        )
        db.session.add(rec)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"failed to store review: {str(e)}"}), 500

    return jsonify({
        "id": rec.id,
        "detected_lang": detected_lang,
        "translated_text": translated if detected_lang != 'en' else None,
        "overall_sentiment": overall,
        "aspects": aspects,
        "created_at": rec.created_at.isoformat()
    })

@app.route('/trends', methods=['GET'])
def trends():
    try:
        days = int(request.args.get('days', 7))
    except Exception:
        days = 7
    since = datetime.utcnow() - timedelta(days=days)

    q = Review.query.filter(Review.created_at >= since).order_by(Review.created_at.asc()).all()
    by_day = defaultdict(lambda: Counter())
    for r in q:
        day = r.created_at.strftime('%Y-%m-%d')
        by_day[day][r.overall_sentiment] += 1

    series = []
    for day in sorted(by_day.keys()):
        counts = by_day[day]
        total = sum(counts.values()) or 1
        series.append({
            "date": day,
            "positive": counts.get('Positive', 0),
            "negative": counts.get('Negative', 0),
            "neutral": counts.get('Neutral', 0),
            "total": total
        })

    return jsonify({
        "days": days,
        "total_reviews": len(q),
        "series": series
    })

if __name__ == '__main__':
    app.run(debug=True)
