import os

# --- CONFIGURATION ---
os.environ['HF_HOME'] = r"D:\Divyansh\huggingface_cache"
# ---------------------

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
from newspaper import Article, Config
from GoogleNews import GoogleNews
import nltk

# Ensure NLTK data is present for scraping
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'veritas-secure-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database & Login Manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- DATABASE MODEL ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# --- LOAD AI MODEL ---
MODEL_PATH = "./model_output"
model, tokenizer, device, explainer = None, None, None, None

if os.path.exists(MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        explainer = SequenceClassificationExplainer(model, tokenizer)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("WARNING: Model not found. Please run train.py first.")

# --- HELPER FUNCTIONS ---

def check_google(query):
    """
    Searches Google News to verify the claim.
    """
    try:
        gn = GoogleNews()
        gn.set_lang('en')
        # Clean and shorten query for better search results
        clean_query = " ".join(query.replace('\n', ' ').split()[:10])
        gn.search(clean_query)
        res = gn.result()
        
        if not res:
            return {"status": "No Matches Found", "sources": []}
        
        trusted_domains = [
            'bbc', 'reuters', 'indiatoday', 'ndtv', 'timesofindia', 
            'cnn', 'thehindu', 'indianexpress', 'isro', 'nasa', 
            'pib', 'who', 'rbi', 'openai', 'techcrunch', 'mint'
        ]
        
        sources = []
        debunked = False
        
        for r in res[:5]:
            src = r.get('media', '') or r.get('desc', '')
            link = r.get('link', '')
            title = r.get('title', '').lower()
            
            # Check if the article itself is a Fact Check
            if "fact check" in title or "fake" in title or "hoax" in title:
                debunked = True
                
            valid = any(t in src.lower() or t in link for t in trusted_domains)
            sources.append({
                "title": r.get('title'), 
                "link": link, 
                "source": src, 
                "valid": valid
            })
            
        if debunked:
            return {"status": "Debunked", "sources": sources}
            
        status = "Verified" if any(s['valid'] for s in sources) else "Unverified"
        return {"status": status, "sources": sources}
    except:
        return {"status": "Error", "sources": []}

def check_known_hoaxes(text):
    """
    Instant fail for known viral fake news patterns.
    """
    text_lower = text.lower()
    
    # 1. UNESCO / National Anthem
    if "unesco" in text_lower and ("national anthem" in text_lower or "jana gana mana" in text_lower):
        return "FAKE", "DETECTED: Viral UNESCO Hoax (UNESCO never declared this)."
        
    # 2. Petrol Ban
    if "ban" in text_lower and "petrol" in text_lower and ("secretly" in text_lower or "fined" in text_lower):
        return "FAKE", "DETECTED: False alarmism about government bans."
        
    # 3. Moon Landing
    if "moon landing" in text_lower and ("hoax" in text_lower or "fake" in text_lower):
        return "FAKE", "DETECTED: Conspiracy theory debunked by space agencies."
        
    # 4. Medical Scams
    if ("doctors" in text_lower and "furious" in text_lower) or ("miracle cure" in text_lower):
        return "FAKE", "DETECTED: Typical health scam pattern."

    # 5. Aliens
    if "aliens" in text_lower and "landed" in text_lower:
        return "FAKE", "DETECTED: Unverified sensationalist claim."

    return None, None

def detect_sensationalism(text):
    """
    Heuristic analysis for clickbait language.
    """
    score = 0
    triggers = []
    keywords = [
        "shocking", "banned", "secret", "miracle", "cure", "hoax", 
        "exposed", "furious", "destroy", "panic", "forward this", "whatsapp"
    ]
    
    text_lower = text.lower()
    for word in keywords:
        if word in text_lower:
            score += 0.25
            triggers.append(word)
    
    if len(text) > 0 and sum(1 for c in text if c.isupper()) / len(text) > 0.2:
        score += 0.2
        triggers.append("Excessive Caps")
        
    return min(score, 0.9), triggers

# --- ROUTES ---

@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user.username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = User.query.filter_by(username=request.form.get('username')).first()
        if u and check_password_hash(u.password, request.form.get('password')):
            login_user(u)
            return redirect(url_for('home'))
        flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if User.query.filter_by(username=request.form.get('username')).first():
            flash('Username already exists.', 'error')
        else:
            new_u = User(username=request.form.get('username'), 
                         password=generate_password_hash(request.form.get('password'), method='pbkdf2:sha256'))
            db.session.add(new_u)
            db.session.commit()
            login_user(new_u)
            return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/check_user/<username>', methods=['GET'])
def check_user(username):
    exists = User.query.filter_by(username=username).first() is not None
    return jsonify({'exists': exists})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model: return jsonify({'error': 'Model not loaded'})
    
    data = request.json
    text = data.get('text', '').strip()
    
    # URL Scraping
    if data.get('url'):
        try:
            conf = Config()
            conf.browser_user_agent = 'Mozilla/5.0'
            conf.request_timeout = 10
            a = Article(data.get('url'), config=conf)
            a.download()
            a.parse()
            text = a.text.strip()
            # Fallback if text is empty
            if len(text) < 50: text = f"{a.title}. {a.meta_description}"
        except Exception as e:
            return jsonify({'error': f'URL Error: {str(e)}'})
    
    if not text: return jsonify({'error': 'No input text found.'})

    # 1. CHECK KNOWN HOAXES (Instant Fail)
    forced_label, forced_note = check_known_hoaxes(text)
    if forced_label == "FAKE":
        return jsonify({
            'label': "FAKE", 
            'confidence': 100.0, 
            'preview': text[:150]+"...", 
            'explanation': [("Pattern Match", 1.0)], 
            'fact_check': {"status": "Debunked", "sources": []}, 
            'note': forced_note
        })

    # 2. AI PREDICTION
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=1)
    
    # Logic: Index 0 = REAL, Index 1 = FAKE (Matches Train.py)
    ai_real = probs[0][0].item()
    ai_fake = probs[0][1].item()
    
    # 3. HEURISTICS ADJUSTMENT
    h_score, triggers = detect_sensationalism(text)
    
    final_fake = min(ai_fake + h_score, 1.0)
    final_real = 1.0 - final_fake
    
    label = "FAKE" if final_fake > 0.5 else "REAL"
    conf = round((final_fake if label=="FAKE" else final_real) * 100, 2)
    
    # 4. FACT CHECK & FINAL DECISION
    fc = check_google(text)
    note = "Analysis Complete"
    
    if fc['status'] == "Verified":
        # Trusted sources confirm it -> It's REAL
        label = "REAL"
        conf = max(95.0, conf)
        note = "Verified Authentic by major news outlets."
        
    elif fc['status'] == "Debunked":
        # Fact checkers say it's false -> It's FAKE
        label = "FAKE"
        conf = 100.0
        note = "CRITICAL: Officially debunked by fact-checkers."
        
    elif label == "REAL" and fc['status'] in ["No Matches Found", "Unverified"]:
        # AI likes it, but no proof found. Be cautious.
        note = "Unverified: AI passed text style, but no external sources found."
        
    elif label == "FAKE":
        note = "High Risk: Suspicious style detected."

    # Explainability
    try:
        exp = sorted(explainer(text[:2000]), key=lambda x: abs(x[1]), reverse=True)[:6]
    except: exp = []

    return jsonify({
        'label': label, 
        'confidence': conf, 
        'preview': text[:150]+"...", 
        'explanation': exp, 
        'fact_check': fc, 
        'note': note
    })

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True, port=5000)