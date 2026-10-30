# VerifiX AI — Intelligent Misinformation Detection & Evidence Verification Engine

VerifiX AI is an evidence-grounded misinformation detection platform powered by Google Gemini and Google Search Grounding. It provides transparent, explainable factual verification by decomposing claims, cross-referencing live web evidence, clustering syndicated wire reporting, detecting genuine evidentiary disputes, and validating source grounding.

---

## 🌟 Key Features

- **Bidirectional Evidence Search**: Searches both supporting and refuting sources to avoid confirmation bias.
- **Claim Decomposition**: Automatically splits long articles or complex assertions into discrete verifiable factual statements.
- **Wire & PR Syndication Clustering**: Recognizes syndicated wire copies (e.g., Associated Press, Reuters, PR Newswire) to prevent duplicate reports from artificially inflating credibility.
- **Conflict & Divergence Discrimination**: Distinguishes genuine evidentiary disputes from harmless methodological or demographic differences.
- **Citation Grounding Validation**: Directly verifies citations against search grounding chunks to eliminate hallucinations.
- **Multi-Key Rate-Limit Fallback**: Automatically switches between multiple API keys (`GEMINI_API_KEY_1` → `GEMINI_API_KEY_2` → `GEMINI_API_KEY_3`) upon encountering rate limits (HTTP 429).
- **Hardened Security Protections**:
  - Full SSRF protection against loopback, private RFC-1918 subnets, cloud metadata endpoints, and DNS rebinding.
  - Strict file-upload MIME validation and temporary file cleanup.
  - Sliding-window IP rate limiting.
  - Defensive prompt-injection quarantine for untrusted user submissions.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, Flask 3.0.3, Werkzeug
- **AI Engine**: Google Gemini 3.8 Flash (`gemini-3.8-flash`) with Google Search Grounding
- **Frontend**: HTML5, Jinja2, Tailwind CSS, Vanilla JavaScript
- **Testing**: Python `unittest`, 100-case reviewed offline evaluation benchmark

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/narshimalucky2-ai/verifix.ai.git
cd verifix.ai
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and configure your Google Gemini API key:
```env
GEMINI_MODEL=gemini-3.8-flash
GEMINI_API_KEY_1=your_gemini_api_key_here
```

### 5. Run the Server
```bash
python main.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## 🧪 Running Automated Tests

Run the full test suite (34 deterministic & integration unit tests):
```bash
python -m unittest discover tests
```

Run the 100-case reviewed evaluation benchmark:
```bash
python tests/evaluate_accuracy.py
```

---

## 📡 API Endpoints

- `GET /` — Web interface
- `GET /api/health` — Service health and active capabilities
- `POST /api/verify` — Fact-checking verification endpoint (accepts text, links, or image uploads)

---

## 🔒 Security & Privacy

- `.env` and credentials are never checked into version control.
- All external URL requests are validated and filtered against SSRF vulnerabilities before fetching.
- Rate limiting protects against denial-of-service and API quota exhaustion.

