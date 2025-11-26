🛡️ Veritas Pro: Advanced AI Fake News Detection System

Veritas Pro is a state-of-the-art hybrid fake news detection system designed to combat misinformation using a combination of Deep Learning, Heuristic Analysis, and Real-Time Fact-Checking.

Unlike traditional models that only analyze writing style, Veritas Pro cross-references claims with live Google News data to verify facts, ensuring high accuracy even on "formal-sounding" fake news.

🚀 Features

🧠 1. Hybrid Intelligence Engine

AI Layer: Uses a fine-tuned DistilBERT transformer model to detect deceptive writing styles.

Heuristic Layer: Instantly flags known viral hoaxes (e.g., "UNESCO National Anthem", "Forward this on WhatsApp").

Fact-Check Layer: Scrapes Google News in real-time to verify claims against reputable sources (BBC, Reuters, ISRO, etc.).

💻 2. Modern Web Dashboard

Dual Input: Support for raw text analysis and direct URL scraping.

Dark Mode: Fully responsive, eye-friendly dark/light theme UI.

Explainability: Highlights "trigger words" that influenced the AI's decision.

🔒 3. Secure User System

Authentication: Secure Login and Signup system using Flask-Login.

Database: SQLite storage for user credentials (hashed) and search history.

History: Saves recent analyses for quick reference.

📂 Project Structure

AI-Fake-News-Detection-V1.0/
│
├── data/                   # Dataset storage (WELFake + Indian News)
├── instance/               # SQLite Database (created after running app)
├── model_output/           # Trained DistilBERT model files
├── static/                 # CSS styles and JavaScript logic
├── templates/              # HTML templates (Login, Signup, Dashboard)
│
├── app.py                  # Flask Backend Application
├── train.py                # Model Training Script
├── requirements.txt        # Python Dependencies
│
└── Visualizations (Generated during training):
    ├── confusion_matrix.png
    ├── metrics_summary.png
    ├── roc_curve.png
    ├── pr_curve.png
    └── training_history.png


🛠️ Installation & Setup

1. Clone the Repository

git clone [https://github.com/Divyansh010605/AI-Fake-News-Detection-V1.0.git](https://github.com/Divyansh010605/AI-Fake-News-Detection-V1.0.git)
cd AI-Fake-News-Detection-V1.0


2. Install Dependencies

Ensure you have Python 3.10 or 3.11 installed.

pip install -r requirements.txt


Note: If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch first.

3. Setup Cache (Optional but Recommended)

The project is configured to use a specific path for caching huge models (to save C: drive space). You can modify the os.environ['HF_HOME'] line in train.py and app.py to change this path.

⚡ Usage

Step 1: Train the Model

Before running the web app, you must train the AI model. This script merges datasets, balances them, and fine-tunes DistilBERT.

python train.py


Time estimate: 20-40 minutes on an RTX 4070.

Step 2: Run the Web App

Start the Flask server to launch the dashboard.

python app.py


Open your browser and go to: http://127.0.0.1:5000

📊 Model Performance

The model was trained on a balanced dataset of ~70,000 articles (merged Global + Indian news).

Metric Score

Accuracy = ~96%

Precision = 0.95

Recall = 0.97

F1-Score = 0.96

🧪 Testing Cases

Try these inputs to test the Hybrid Engine:

Real: "OpenAI has officially released GPT-4 Turbo with a larger context window." -> Result: REAL (Verified)

Fake (Viral): "UNESCO has declared the Indian National Anthem as the best in the world." -> Result: FAKE (Caught by Heuristic Trap)

Fake (Health): "Doctors are furious! This simple kitchen ingredient cures diabetes overnight." -> Result: FAKE (Caught by AI + No Verification)

