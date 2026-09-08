"""
AI Interview Simulator — Prototype (Step 1-5, working Streamlit version)
----------------------------------------------------
Text-based Q&A loop using Groq + gTTS voice, with per-answer video
recording via plain-HTML MediaRecorder (records real video/audio in
the browser, then you download the clip and upload it back in).

We previously tried a custom Streamlit component (declare_component)
that sent the recording straight to Python with no manual step, but
that hit two platform limitations on Streamlit Community Cloud: the
component's iframe blocked camera/mic permission, and its frontend
files sometimes failed to load reliably once deployed ("trouble
loading the main2.video_recorder component"). Neither is fixable from
our code, so this version — confirmed to reliably open the camera and
record real video — is the one to build on going forward.

Setup:
    pip install streamlit>=1.32 groq gTTS openai-whisper deep-translator
    # Whisper also needs the ffmpeg binary on your system PATH:
    #   Mac:     brew install ffmpeg
    #   Ubuntu:  sudo apt install ffmpeg
    #   Windows: winget install ffmpeg   (or download from ffmpeg.org)

Run:
    streamlit run main2.py

You'll need a free Groq API key from https://console.groq.com
Set it as an environment variable before running (or as a Streamlit secret):
    export GROQ_API_KEY="your_key_here"      (Mac/Linux)
    setx GROQ_API_KEY "your_key_here"         (Windows)

Note on "no other tabs" / proctoring lockdown:
    No website JavaScript can truly prevent a person from opening a new
    browser tab, switching apps, or using Alt+Tab — that control belongs
    to the browser/OS, not the page. What this file does instead is the
    strongest thing actually achievable from a web page: it detects a
    tab switch the instant it happens, can end the interview immediately
    in "Strict mode", requests fullscreen + (where the browser supports
    it) Keyboard Lock to swallow common shortcuts, and warns before the
    tab is closed/reloaded. Treat this as a deterrent + audit trail, not
    an unbreakable lock.
"""

import base64
import io
import json
import os
import random
import tempfile
import streamlit as st
import whisper
from groq import Groq
from gtts import gTTS
from deep_translator import GoogleTranslator

# ---------- CONFIG ----------
APP_NAME = "InterviewAI"
APP_TAGLINE = "Practice smarter. Interview better."
GROQ_MODEL = "openai/gpt-oss-20b"  # fast + good quality on Groq's free tier
QUESTIONS_PER_INTERVIEW = 5  # default/fallback question count
WHISPER_MODEL_SIZE = "base"  # small + fast enough for CPU, decent accuracy

# ---------- SETUP-SCREEN OPTIONS ----------
EXPERIENCE_LEVELS = ["Beginner", "Intermediate", "Advanced"]
INTERVIEW_TYPES = ["Technical", "HR", "Behavioral", "Mixed"]
QUESTION_COUNT_OPTIONS = [5, 10, 15]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

# ---------- PROCTORING CONFIG ----------
DEFAULT_STRICT_MODE = True            # if True, leaving the tab even once ends the interview immediately
MAX_ALLOWED_SCREEN_SWITCHES = 3       # (non-strict mode only) switches tolerated before a warning is escalated
FACE_MISSING_THRESHOLD_MS = 5000      # how long a face must be absent before it's logged
GAZE_AWAY_THRESHOLD_MS = 4000         # how long attention must be turned away before it's logged
MULTI_FACE_THRESHOLD_MS = 2000        # how long multiple faces must persist before it's logged
PHONE_DETECT_COOLDOWN_MS = 8000       # minimum gap between repeated "phone detected" events

# Display name -> language code. Codes work for both deep_translator
# (Google Translate) and gTTS, so the same code drives both translation
# and speech.
LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "French": "fr",
    "Spanish": "es",
}


# ---------- QUESTION BANK ----------
# Each entry: (question_text, difficulty, category). category is one of
# "Technical", "Behavioral", "Situational", "HR". Technical questions are
# grouped under role keywords; a question is offered to a role if any of
# its keywords appear in the candidate's typed role/topic (case-insensitive).
# "generic" technical questions are offered to every role as a fallback so
# uncommon roles still get a reasonable technical mix.
TECHNICAL_QUESTION_BANK = {
    "generic": [
        ("Walk me through how you would approach debugging a production issue you've never seen before.", "Easy"),
        ("What does the term 'technical debt' mean, and how do you decide when to pay it down?", "Easy"),
        ("Explain the difference between unit tests, integration tests, and end-to-end tests.", "Easy"),
        ("How would you design a system to be both scalable and maintainable?", "Medium"),
        ("Describe a time you had to learn a new technology quickly for a project.", "Medium"),
        ("What trade-offs do you consider when choosing between build vs. buy for a tool?", "Medium"),
        ("How do you approach code reviews, both giving and receiving feedback?", "Medium"),
        ("Design a rate limiter for a public API. What approach would you take and why?", "Hard"),
        ("How would you diagnose a memory leak in a long-running service?", "Hard"),
        ("Explain how you'd design a system for high availability with minimal downtime during deploys.", "Hard"),
    ],
    "python": [
        ("What is the difference between a list and a tuple in Python?", "Easy"),
        ("Explain how Python's garbage collection works.", "Medium"),
        ("What is the Global Interpreter Lock (GIL) and how does it affect multithreaded Python programs?", "Medium"),
        ("How would you optimize a slow Python script that processes a large CSV file?", "Hard"),
        ("Explain the difference between multiprocessing and multithreading in Python, and when you'd use each.", "Hard"),
    ],
    "backend": [
        ("What is the difference between SQL and NoSQL databases, and when would you choose each?", "Easy"),
        ("Explain what a REST API is and what makes an API RESTful.", "Easy"),
        ("How do you handle authentication and authorization in a backend service?", "Medium"),
        ("Describe how you'd design a database schema for an e-commerce order system.", "Medium"),
        ("How would you design a system to handle 10,000 requests per second?", "Hard"),
        ("Explain database indexing and how it affects query performance and write speed.", "Hard"),
    ],
    "frontend": [
        ("What is the difference between 'let', 'const', and 'var' in JavaScript?", "Easy"),
        ("Explain the concept of the virtual DOM and why it's used.", "Easy"),
        ("How do you optimize the performance of a web page with a large amount of rendered data?", "Medium"),
        ("Explain how you'd manage state in a large single-page application.", "Medium"),
        ("How would you diagnose and fix a memory leak in a long-running single-page application?", "Hard"),
    ],
    "react": [
        ("What is the difference between state and props in React?", "Easy"),
        ("Explain the purpose of React hooks like useEffect and useMemo.", "Medium"),
        ("How would you optimize re-renders in a React application with deeply nested components?", "Hard"),
    ],
    "data": [
        ("What is the difference between supervised and unsupervised learning?", "Easy"),
        ("Explain overfitting and how you would detect and address it.", "Medium"),
        ("How would you handle a dataset with significant class imbalance?", "Medium"),
        ("Walk me through how you would evaluate whether a machine learning model is production-ready.", "Hard"),
        ("Explain the bias-variance trade-off and how it guides model selection.", "Hard"),
    ],
    "devops": [
        ("What is the difference between continuous integration and continuous deployment?", "Easy"),
        ("Explain what containers are and how they differ from virtual machines.", "Easy"),
        ("How would you design a CI/CD pipeline for a microservices architecture?", "Medium"),
        ("How would you troubleshoot a Kubernetes pod that keeps crashing on startup?", "Hard"),
    ],
    "java": [
        ("What is the difference between an abstract class and an interface in Java?", "Easy"),
        ("Explain how garbage collection works in the JVM.", "Medium"),
        ("How would you diagnose a thread-safety issue in a multi-threaded Java application?", "Hard"),
    ],
}

# Maps role keywords to the technical buckets above, so e.g. "Python Backend
# Developer" pulls from both "python" and "backend" in addition to "generic".
ROLE_KEYWORD_MAP = {
    "python": "python",
    "django": "python",
    "flask": "python",
    "backend": "backend",
    "back-end": "backend",
    "api": "backend",
    "server": "backend",
    "frontend": "frontend",
    "front-end": "frontend",
    "javascript": "frontend",
    "react": "react",
    "vue": "frontend",
    "angular": "frontend",
    "data scientist": "data",
    "data science": "data",
    "machine learning": "data",
    "ml engineer": "data",
    "devops": "devops",
    "sre": "devops",
    "cloud": "devops",
    "kubernetes": "devops",
    "java": "java",
}

BEHAVIORAL_QUESTION_BANK = [
    ("Tell me about a time you disagreed with a teammate. How did you handle it?", "Easy"),
    ("Describe a project you're particularly proud of and why.", "Easy"),
    ("Tell me about a time you missed a deadline. What happened and what did you learn?", "Medium"),
    ("Describe a situation where you had to give difficult feedback to a colleague.", "Medium"),
    ("Tell me about a time you had to work with incomplete or changing requirements.", "Medium"),
    ("Describe a time you failed at something important. How did you recover?", "Hard"),
    ("Tell me about a time you had to influence a decision without having formal authority.", "Hard"),
]

SITUATIONAL_QUESTION_BANK = [
    ("If you discovered a bug in production right before a major release, what would you do?", "Easy"),
    ("How would you handle being assigned two urgent tasks with the same deadline?", "Medium"),
    ("If a stakeholder kept changing requirements midway through a project, how would you respond?", "Medium"),
    ("Suppose your manager asked you to ship a feature you believed was technically unsound. What would you do?", "Hard"),
    ("If you noticed a teammate was consistently underperforming, how would you approach the situation?", "Hard"),
]

HR_QUESTION_BANK = [
    ("Why are you interested in this role?", "Easy"),
    ("Where do you see yourself in the next few years?", "Easy"),
    ("What are your salary expectations for this position?", "Medium"),
    ("Why are you considering leaving your current role?", "Medium"),
    ("How do you prioritize your work-life balance while meeting tight deadlines?", "Hard"),
]


def _matching_technical_buckets(role: str) -> list:
    """Return the technical-question bucket keys relevant to a typed role,
    always including 'generic' as a baseline."""
    role_lower = role.lower()
    buckets = {"generic"}
    for keyword, bucket in ROLE_KEYWORD_MAP.items():
        if keyword in role_lower:
            buckets.add(bucket)
    return list(buckets)


def pick_interview_questions(
    role: str,
    n: int = QUESTIONS_PER_INTERVIEW,
    interview_type: str = "Mixed",
    difficulty: str = "Medium",
) -> list:
    """Randomly build a non-repeating set of n questions for this interview,
    mixing technical, behavioral, situational, and HR questions. Technical
    questions are drawn from buckets relevant to the typed role/topic.

    interview_type narrows the category mix ("Technical" / "HR" /
    "Behavioral" / "Mixed"); "Behavioral" also pulls in situational
    questions since they're closely related. difficulty is used as a soft
    preference — matching questions are tried first, then the pool is
    topped up with other difficulties so a short/niche role never comes up
    short on questions."""
    pool = []
    for bucket in _matching_technical_buckets(role):
        for text, diff in TECHNICAL_QUESTION_BANK.get(bucket, []):
            pool.append({"text": text, "difficulty": diff, "category": "Technical"})
    for text, diff in BEHAVIORAL_QUESTION_BANK:
        pool.append({"text": text, "difficulty": diff, "category": "Behavioral"})
    for text, diff in SITUATIONAL_QUESTION_BANK:
        pool.append({"text": text, "difficulty": diff, "category": "Situational"})
    for text, diff in HR_QUESTION_BANK:
        pool.append({"text": text, "difficulty": diff, "category": "HR"})

    # De-duplicate (a question could theoretically appear via multiple buckets)
    seen_texts = set()
    unique_pool = []
    for q in pool:
        if q["text"] not in seen_texts:
            seen_texts.add(q["text"])
            unique_pool.append(q)

    # Narrow by interview type first.
    if interview_type == "Technical":
        typed_pool = [q for q in unique_pool if q["category"] == "Technical"]
    elif interview_type == "HR":
        typed_pool = [q for q in unique_pool if q["category"] == "HR"]
    elif interview_type == "Behavioral":
        typed_pool = [q for q in unique_pool if q["category"] in ("Behavioral", "Situational")]
    else:  # "Mixed"
        typed_pool = unique_pool
    if not typed_pool:  # safety net if a narrow filter leaves nothing
        typed_pool = unique_pool

    # Prefer the requested difficulty, then fill any remaining slots from
    # the rest of the (type-filtered) pool so we always return n questions.
    preferred = [q for q in typed_pool if q["difficulty"] == difficulty]
    remaining = [q for q in typed_pool if q["difficulty"] != difficulty]
    random.shuffle(preferred)
    random.shuffle(remaining)
    ordered = preferred + remaining

    n = min(n, len(ordered))
    return ordered[:n]


def translate_text(text: str, target_lang_code: str) -> str:
    """Translate text into the selected language via Google Translate
    (through deep_translator, free, no API key needed). Falls back to the
    original English text if translation fails for any reason, so a
    translation hiccup never breaks the interview."""
    if not text or target_lang_code == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang_code).translate(text)
    except Exception:
        return text


@st.cache_resource
def load_whisper_model():
    """Loaded once and cached across reruns/users — downloading the model
    weights every rerun would be far too slow."""
    return whisper.load_model(WHISPER_MODEL_SIZE)

# ---------- SETUP ----------
st.set_page_config(page_title=f"{APP_NAME} — AI Interview Simulator", page_icon="🎤", layout="wide")


def inject_theme():
    """Global CSS for a clean, professional SaaS-style look. Pure CSS only —
    no extra Python packages — so it can't introduce a deployment problem."""
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}

        .ia-navbar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.6rem 0 1.4rem 0; border-bottom: 1px solid #e7e9ee; margin-bottom: 1.6rem;
        }
        .ia-navbar .ia-logo {
            font-size: 1.35rem; font-weight: 800; color: #1a1f36; letter-spacing: -0.02em;
        }
        .ia-navbar .ia-logo span { color: #4f46e5; }
        .ia-badge {
            background: #eef2ff; color: #4338ca; padding: 4px 12px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600;
        }

        .ia-hero {
            text-align: center; padding: 2.2rem 1rem 1.4rem 1rem;
        }
        .ia-hero h1 {
            font-size: 2.6rem; font-weight: 800; color: #111827; margin-bottom: 0.4rem; letter-spacing: -0.03em;
        }
        .ia-hero h1 span { color: #4f46e5; }
        .ia-hero p.tagline {
            font-size: 1.25rem; font-weight: 600; color: #4f46e5; margin-bottom: 0.6rem;
        }
        .ia-hero p.desc {
            font-size: 1.02rem; color: #4b5563; max-width: 640px; margin: 0 auto 1.6rem auto; line-height: 1.55;
        }

        .ia-card {
            background: #ffffff; border: 1px solid #eceef2; border-radius: 16px;
            padding: 1.4rem 1.3rem; height: 100%;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        .ia-card:hover { box-shadow: 0 6px 18px rgba(16, 24, 40, 0.08); transform: translateY(-2px); }
        .ia-card .ia-icon { font-size: 1.7rem; margin-bottom: 0.5rem; }
        .ia-card h4 { font-size: 1.02rem; font-weight: 700; color: #1a1f36; margin: 0 0 0.3rem 0; }
        .ia-card p { font-size: 0.9rem; color: #667085; margin: 0; line-height: 1.45; }

        .ia-stat { text-align: center; padding: 0.6rem 0; }
        .ia-stat .ia-stat-num { font-size: 1.9rem; font-weight: 800; color: #4f46e5; }
        .ia-stat .ia-stat-label { font-size: 0.85rem; color: #667085; font-weight: 500; }

        .ia-summary {
            background: #f8f9fc; border: 1px solid #eceef2; border-radius: 14px;
            padding: 1.1rem 1.3rem; margin: 0.8rem 0 1.2rem 0;
        }
        .ia-summary h4 { margin-top: 0; color: #1a1f36; }
        .ia-summary .ia-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.92rem; }
        .ia-summary .ia-row b { color: #1a1f36; }

        div.stButton > button {
            border-radius: 10px; font-weight: 600; padding: 0.55rem 1.4rem;
        }
        div.stButton > button[kind="primary"] {
            background: #4f46e5; border-color: #4f46e5;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #4338ca; border-color: #4338ca;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_navbar(subtitle: str = ""):
    st.markdown(
        f"""
        <div class="ia-navbar">
            <div class="ia-logo">Interview<span>AI</span></div>
            <div class="ia-badge">{subtitle or "AI Mock Interview Platform"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Add GROQ_API_KEY to Streamlit Secrets or set it as an environment variable.")
    st.stop()

client = Groq(api_key=api_key)

# ---------- SESSION STATE ----------
# session_state persists data across reruns (Streamlit reruns the whole
# script on every interaction, so this is where we keep the conversation).
if "stage" not in st.session_state:
    st.session_state.stage = "landing"        # landing -> setup -> interview -> feedback
if "role" not in st.session_state:
    st.session_state.role = ""
if "language" not in st.session_state:
    st.session_state.language = "en"  # language code chosen on the setup screen
if "experience_level" not in st.session_state:
    st.session_state.experience_level = "Intermediate"
if "interview_type" not in st.session_state:
    st.session_state.interview_type = "Mixed"
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Medium"
if "num_questions" not in st.session_state:
    st.session_state.num_questions = QUESTIONS_PER_INTERVIEW
if "history" not in st.session_state:
    st.session_state.history = []             # list of {"question": ..., "answer": ...}
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "spoken_question" not in st.session_state:
    st.session_state.spoken_question = ""      # tracks which question we've already voiced
if "translated_question" not in st.session_state:
    st.session_state.translated_question = ""  # question text in the selected display language
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "recorded_video" not in st.session_state:
    st.session_state.recorded_video = None  # holds the uploaded video clip for the current question
if "transcribed_answer" not in st.session_state:
    st.session_state.transcribed_answer = ""  # Whisper's transcription of the current answer
if "reaction_audio" not in st.session_state:
    st.session_state.reaction_audio = None  # spoken version of the last reaction
if "question_queue" not in st.session_state:
    st.session_state.question_queue = []  # remaining bank questions for this interview
if "proctor_log" not in st.session_state:
    st.session_state.proctor_log = []  # synced copy of client-side proctoring events
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False
if "fullscreen_prompt_shown" not in st.session_state:
    st.session_state.fullscreen_prompt_shown = False
if "switch_limit_warned" not in st.session_state:
    st.session_state.switch_limit_warned = False
if "strict_mode" not in st.session_state:
    st.session_state.strict_mode = DEFAULT_STRICT_MODE
if "disqualified" not in st.session_state:
    st.session_state.disqualified = False
if "disqualify_reason" not in st.session_state:
    st.session_state.disqualify_reason = ""


# ---------- GROQ HELPERS ----------
def generate_question(role: str, history: list) -> str:
    """Ask Groq for exactly one interview question based on role + prior Q&A."""
    context_lines = [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
    context = "\n\n".join(context_lines) if context_lines else "No prior questions yet."

    prompt = f"""You are a professional technical interviewer.
Candidate's target role: {role}

Conversation so far:
{context}

Generate exactly ONE single-sentence interview question that follows naturally
from the conversation so far (or starts the interview if there is none yet).
Output ONLY the question text — no preamble, no numbering, no explanation."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def generate_feedback(role: str, history: list) -> str:
    """Ask Groq to evaluate the full interview transcript."""
    transcript = "\n\n".join(
        f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer']}" for i, h in enumerate(history)
    )

    prompt = f"""You are an expert interview coach.
Candidate's target role: {role}

Full interview transcript:
{transcript}

Give concise, constructive feedback covering:
- Clarity of answers
- Relevance to the questions asked
- Depth/detail of responses
- 2-3 concrete suggestions to improve

Keep it to a short paragraph plus a bullet list of suggestions."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def generate_reaction(role: str, question: str, answer: str) -> dict:
    """Ask Groq to score this single answer (1-10) and give a short,
    spoken-style reaction — this is what makes the interview feel live
    and back-and-forth instead of silent until the very end."""
    prompt = f"""You are a professional technical interviewer conducting a live interview.
Candidate's target role: {role}

You just asked: {question}
The candidate answered: {answer}

Respond with ONLY a JSON object in exactly this format, no other text:
{{"score": <integer 1-10>, "reaction": "<one short, natural, spoken-style sentence reacting to this answer, as an interviewer would say out loud>"}}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    raw = response.choices[0].message.content.strip()

    try:
        # Models sometimes wrap JSON in ```json fences despite instructions —
        # strip those before parsing.
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        score = int(data.get("score", 5))
        reaction = str(data.get("reaction", "Thanks for that answer.")).strip()
        score = max(1, min(10, score))  # clamp to 1-10 in case the model drifts
        return {"score": score, "reaction": reaction}
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback so a rare malformed response never crashes the app
        return {"score": 5, "reaction": "Thanks for that answer, let's move on."}


def generate_final_report(role: str, history: list) -> dict:
    """Ask Groq for a multi-dimension score breakdown plus strengths/
    improvements, covering interview performance only (proctoring events
    are summarized separately, never mixed into this report)."""
    transcript = "\n\n".join(
        f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer']}" for i, h in enumerate(history)
    )
    prompt = f"""You are an expert interview panel evaluating a candidate.
Candidate's target role: {role}

Full interview transcript:
{transcript}

Respond with ONLY a JSON object in exactly this format, no other text:
{{
  "technical_score": <integer 1-10>,
  "communication_score": <integer 1-10>,
  "confidence_score": <integer 1-10>,
  "relevance_score": <integer 1-10>,
  "overall_score": <integer 1-10>,
  "strengths": ["short strength 1", "short strength 2"],
  "improvements": ["short improvement 1", "short improvement 2"]
}}"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    raw = response.choices[0].message.content.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    def _clamp(v):
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 5

    try:
        data = json.loads(cleaned)
        return {
            "technical_score": _clamp(data.get("technical_score", 5)),
            "communication_score": _clamp(data.get("communication_score", 5)),
            "confidence_score": _clamp(data.get("confidence_score", 5)),
            "relevance_score": _clamp(data.get("relevance_score", 5)),
            "overall_score": _clamp(data.get("overall_score", 5)),
            "strengths": [str(s) for s in data.get("strengths", [])][:5],
            "improvements": [str(s) for s in data.get("improvements", [])][:5],
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        scores = [h["score"] for h in history if "score" in h]
        avg = round(sum(scores) / len(scores)) if scores else 5
        return {
            "technical_score": avg, "communication_score": avg,
            "confidence_score": avg, "relevance_score": avg, "overall_score": avg,
            "strengths": [], "improvements": [],
        }


# ---------- VOICE HELPER ----------
def text_to_speech(text: str, lang_code: str = "en") -> bytes:
    """Convert text into spoken audio (in-memory, no temp files needed).
    lang_code should match the already-translated text passed in."""
    try:
        tts = gTTS(text=text, lang=lang_code)
    except ValueError:
        # gTTS doesn't support every language code — fall back to English
        # speech rather than crashing the app.
        tts = gTTS(text=text, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


# ---------- AVATAR + SPEAKING INDICATOR ----------
def render_avatar_with_speech(audio_bytes: bytes, unique_id: str, auto_advance: bool = False):
    """Shows a simple SVG interviewer face with a glowing ring that pulses
    exactly while the audio is playing, then goes still when it ends —
    this is what gives the 'someone is actually talking to you' feel,
    without needing real lip-sync (which isn't realistic to build for
    free within this timeline).

    If auto_advance=True, the moment the audio finishes, the browser
    navigates itself (via a query param) so the app moves to the next
    step with NO click needed — used after the AI's reaction, so the
    interview keeps moving on its own once it's done talking."""
    audio_b64 = base64.b64encode(audio_bytes).decode()
    ended_message = "Waiting for your answer..." if not auto_advance else "Moving on..."
    # Piggyback the client-side proctoring event log (kept on window.top by
    # render_proctoring_system) onto this same navigation, so it reaches
    # Python without needing a separate round trip.
    # Wrapped in try/catch: this component runs inside a sandboxed iframe,
    # and browsers can block a script-driven (non-click) top-level
    # navigation from such a frame, especially if the tab has lost focus
    # (e.g. the candidate switched tabs). Without this guard, that block
    # surfaces as an uncaught JS error. The manual button below the
    # avatar is the fallback if auto-advance can't fire for that reason.
    auto_advance_js = (
        """
        try {
            var __events = (window.top.__proctorState && window.top.__proctorState.events) ? window.top.__proctorState.events : [];
            window.top.location.search = "?advance=""" + unique_id + """&pevents=" + encodeURIComponent(JSON.stringify(__events));
        } catch (e) {
            console.warn("Auto-advance navigation blocked; use the manual button instead.", e);
        }
        """
        if auto_advance else ""
    )

    st.components.v1.html(
        f"""
        <div style="font-family: sans-serif; text-align:center;">
          <div id="avatar-{unique_id}" style="
              width:140px; height:140px; margin:0 auto;
              border-radius:50%; background:#2b2f38;
              display:flex; align-items:center; justify-content:center;
              transition: box-shadow 0.15s ease-in-out;
          ">
            <svg width="90" height="90" viewBox="0 0 100 100">
              <circle cx="50" cy="38" r="20" fill="#e0c9a6"/>
              <path d="M20 95 Q50 60 80 95 Z" fill="#4a5568"/>
              <circle cx="42" cy="36" r="3" fill="#2b2f38"/>
              <circle cx="58" cy="36" r="3" fill="#2b2f38"/>
              <path d="M42 48 Q50 54 58 48" stroke="#2b2f38" stroke-width="2" fill="none" stroke-linecap="round"/>
            </svg>
          </div>
          <p id="status-{unique_id}" style="color:gray; margin-top:8px;">🔊 Speaking...</p>
          <audio id="player-{unique_id}" autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
          </audio>
        </div>

        <style>
          @keyframes pulse-{unique_id} {{
            0%   {{ box-shadow: 0 0 0 0 rgba(66, 153, 225, 0.6); }}
            70%  {{ box-shadow: 0 0 0 18px rgba(66, 153, 225, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(66, 153, 225, 0); }}
          }}
          .talking-{unique_id} {{
            animation: pulse-{unique_id} 1.1s infinite;
          }}
        </style>

        <script>
          const audioEl = document.getElementById("player-{unique_id}");
          const avatarEl = document.getElementById("avatar-{unique_id}");
          const statusEl = document.getElementById("status-{unique_id}");

          audioEl.addEventListener("play", () => {{
            avatarEl.classList.add("talking-{unique_id}");
            statusEl.innerText = "🔊 Speaking...";
          }});
          audioEl.addEventListener("pause", () => {{
            avatarEl.classList.remove("talking-{unique_id}");
          }});
          audioEl.addEventListener("ended", () => {{
            avatarEl.classList.remove("talking-{unique_id}");
            statusEl.innerText = "{ended_message}";
            {auto_advance_js}
          }});
        </script>
        """,
        height=230,
    )


# ---------- TRANSCRIPTION HELPER ----------
def transcribe_video(uploaded_file) -> str:
    """Save the uploaded clip to a temp file and run Whisper on it.
    Whisper (via ffmpeg under the hood) extracts the audio track itself,
    so we don't need to separately strip audio from the video."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        model = load_whisper_model()
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    finally:
        os.remove(tmp_path)


# ---------- PROCTORING SYSTEM ----------
def render_proctoring_system(strict_mode: bool = False):
    """Injects a persistent proctoring overlay + monitoring logic into the
    TOP browser window (not this disposable iframe). Streamlit reruns
    recreate this component's iframe on every interaction, but window.top
    is the actual browser tab and survives reruns, so state stashed there
    (camera stream, event log, timers) persists across the whole interview.
    This mirrors the window.top.location.search trick already used in
    render_avatar_with_speech above for the same reason: plain
    st.components.v1.html has no real backend channel, so window.top is
    used as the shared, persistent place to keep proctoring state.

    Honesty notes (do not remove):
    - This can only tell that the candidate left the interview tab/window,
      never what other tab or app they went to. Browsers don't expose that.
    - Face/gaze/phone detection are heuristic, browser-only computer vision
      running on the candidate's device. They flag *possible* issues for
      review, not confirmed cheating.
    """
    st.components.v1.html(
        f"""
        <script>
        (function() {{
            // If this frame can't safely reach the top window (blocked by
            // browser sandboxing — e.g. the app is embedded elsewhere, or a
            // popup/new-tab context) bail out quietly instead of throwing.
            // An uncaught error here previously bubbled up as a visible
            // page error whenever a new tab/page was opened during the
            // interview.
            try {{ void window.top.document; }} catch (e) {{
                console.warn("Proctoring disabled: top window is not accessible from this frame.", e);
                return;
            }}

            const MAX_SWITCHES = {MAX_ALLOWED_SCREEN_SWITCHES};
            const FACE_MISSING_MS = {FACE_MISSING_THRESHOLD_MS};
            const GAZE_MS = {GAZE_AWAY_THRESHOLD_MS};
            const MULTI_FACE_MS = {MULTI_FACE_THRESHOLD_MS};
            const PHONE_COOLDOWN_MS = {PHONE_DETECT_COOLDOWN_MS};
            const STRICT_MODE = {"true" if strict_mode else "false"};

            // Re-run every mount so a strict-mode change (e.g. a fresh
            // interview started with a different setting) takes effect,
            // but never install the heavy camera/model-loading logic twice.
            window.top.__proctorStrictMode = STRICT_MODE;
            if (window.top.__proctorInit) return;  // already running, nothing to do
            window.top.__proctorInit = true;
            window.top.addEventListener("error", function(evt) {{
                // Swallow errors from this proctoring layer so a detection
                // glitch (e.g. a CDN model failing to load) never shows up
                // as a page-level error to the candidate.
                console.warn("Proctoring script error (ignored):", evt.message);
            }});
            window.top.__proctorState = {{
                events: [], screenSwitches: 0, startTime: Date.now(),
                faceAbsentSince: null, multiFaceSince: null, lookAwaySince: null,
                phoneLastSeen: 0, stream: null, hiddenAt: null, violationSent: false
            }};

            // Best-effort deterrents. None of these can stop a determined
            // person from opening another tab (that's outside any page's
            // control) — they only raise friction and record intent.
            window.top.addEventListener('beforeunload', function(e) {{
                e.preventDefault();
                e.returnValue = '';
            }});
            doc.addEventListener('contextmenu', function(e) {{ e.preventDefault(); }});
            doc.addEventListener('keydown', function(e) {{
                const blockCombo = (e.key === 'F12') ||
                    (e.ctrlKey && e.shiftKey && ['I', 'J', 'C'].includes(e.key)) ||
                    (e.ctrlKey && e.key === 'u') ||
                    (e.metaKey && e.altKey && ['I', 'J', 'C'].includes(e.key));
                if (blockCombo) e.preventDefault();
            }});

            const doc = window.top.document;
            const overlay = doc.createElement('div');
            overlay.id = 'proctor-overlay';
            overlay.style.cssText = 'position:fixed;top:10px;right:10px;width:230px;z-index:999999;' +
                'font-family:sans-serif;background:#111;color:#eee;padding:10px;border-radius:8px;' +
                'font-size:12px;max-height:75vh;overflow:auto;box-shadow:0 2px 10px rgba(0,0,0,0.4);';
            overlay.innerHTML =
                '<video id="proctor-video" autoplay muted playsinline ' +
                'style="width:100%;border-radius:6px;background:#000;"></video>' +
                '<div id="proctor-status" style="margin-top:6px;color:#9ae6b4;">Starting proctoring...</div>' +
                '<div style="margin-top:6px;font-weight:bold;">Proctoring Events (<span id="proctor-count">0</span>)</div>' +
                '<div id="proctor-log" style="max-height:170px;overflow:auto;margin-top:4px;"></div>';
            doc.body.appendChild(overlay);

            const videoEl = doc.getElementById('proctor-video');
            const statusEl = doc.getElementById('proctor-status');
            const logEl = doc.getElementById('proctor-log');
            const countEl = doc.getElementById('proctor-count');

            function addEvent(type, severity, description) {{
                const elapsed = Math.round((Date.now() - window.top.__proctorState.startTime) / 1000);
                window.top.__proctorState.events.push({{
                    type: type, severity: severity, description: description,
                    elapsed_seconds: elapsed, timestamp: new Date().toISOString()
                }});
                countEl.innerText = window.top.__proctorState.events.length;
                const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const ss = String(elapsed % 60).padStart(2, '0');
                const color = severity === 'High' ? '#feb2b2' : (severity === 'Medium' ? '#fbd38d' : '#cbd5e0');
                const row = doc.createElement('div');
                row.style.cssText = 'border-bottom:1px solid #333;padding:3px 0;color:' + color + ';';
                row.innerText = mm + ':' + ss + ' — ' + description + ' (' + severity + ')';
                logEl.insertBefore(row, logEl.firstChild);
            }}

            // ---- camera preview ----
            navigator.mediaDevices.getUserMedia({{ video: true, audio: false }}).then(function(stream) {{
                window.top.__proctorState.stream = stream;
                videoEl.srcObject = stream;
                statusEl.innerText = 'Camera active — monitoring';
            }}).catch(function() {{
                statusEl.innerText = 'Camera permission denied — face/phone checks unavailable';
            }});

            // ---- tab / window switch detection (Page Visibility API) ----
            // We can only observe two moments: the instant this tab is
            // hidden, and the instant it becomes visible again. There is no
            // way to act *while* hidden (the page isn't running), so in
            // strict mode the violation is reported as soon as the
            // candidate comes back — that's the earliest this page can
            // possibly know or do anything about it.
            doc.addEventListener('visibilitychange', function() {{
                const st_ = window.top.__proctorState;
                if (doc.hidden) {{
                    st_.screenSwitches += 1;
                    st_.hiddenAt = Date.now();
                    addEvent('Tab switched', 'Medium', 'Candidate switched tabs or minimized the window');
                }} else if (st_.hiddenAt && STRICT_MODE && !st_.violationSent) {{
                    st_.violationSent = true;
                    try {{
                        window.top.location.search = "?violation=1&pevents=" +
                            encodeURIComponent(JSON.stringify(st_.events));
                    }} catch (e) {{
                        console.warn("Could not report strict-mode violation.", e);
                    }}
                }}
            }});

            // ---- fullscreen exit detection ----
            doc.addEventListener('fullscreenchange', function() {{
                if (!doc.fullscreenElement) {{
                    addEvent('Fullscreen exited', 'Medium', 'Candidate exited fullscreen mode');
                }}
            }});

            // ---- face presence / multi-face / basic gaze via face-api.js ----
            const faceScript = doc.createElement('script');
            faceScript.src = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js';
            faceScript.onload = function() {{
                const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
                const faceapi = window.top.faceapi;
                Promise.all([
                    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
                    faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL)
                ]).then(function() {{
                    setInterval(async function() {{
                        if (!videoEl.srcObject || videoEl.readyState < 2) return;
                        let detections;
                        try {{
                            detections = await faceapi.detectAllFaces(videoEl, new faceapi.TinyFaceDetectorOptions())
                                .withFaceLandmarks(true);
                        }} catch (e) {{ return; }}
                        const now = Date.now();
                        const st_ = window.top.__proctorState;
                        if (detections.length === 0) {{
                            if (!st_.faceAbsentSince) st_.faceAbsentSince = now;
                            else if (now - st_.faceAbsentSince > FACE_MISSING_MS) {{
                                addEvent('Face not detected', 'High', 'No face visible in camera for an extended period');
                                st_.faceAbsentSince = now;
                            }}
                            st_.multiFaceSince = null;
                            st_.lookAwaySince = null;
                        }} else {{
                            st_.faceAbsentSince = null;
                            if (detections.length > 1) {{
                                if (!st_.multiFaceSince) st_.multiFaceSince = now;
                                else if (now - st_.multiFaceSince > MULTI_FACE_MS) {{
                                    addEvent('Multiple faces detected', 'High', detections.length + ' faces visible in frame');
                                    st_.multiFaceSince = now;
                                }}
                            }} else {{
                                st_.multiFaceSince = null;
                                const lm = detections[0].landmarks;
                                const nose = lm.getNose()[3];
                                const leftEye = lm.getLeftEye();
                                const rightEye = lm.getRightEye();
                                const eyeMidX = (leftEye[0].x + rightEye[3].x) / 2;
                                const box = detections[0].detection.box;
                                const normOffset = (nose.x - eyeMidX) / box.width;
                                if (Math.abs(normOffset) > 0.18) {{
                                    if (!st_.lookAwaySince) st_.lookAwaySince = now;
                                    else if (now - st_.lookAwaySince > GAZE_MS) {{
                                        addEvent('Attention directed away from screen', 'Medium',
                                            'Head turned / gaze away from camera for an extended period');
                                        st_.lookAwaySince = now;
                                    }}
                                }} else {{
                                    st_.lookAwaySince = null;
                                }}
                            }}
                        }}
                    }}, 1000);
                    statusEl.innerText = 'Camera active — face monitoring on';
                }}).catch(function() {{
                    statusEl.innerText = 'Camera active — face model failed to load';
                }});
            }};
            doc.head.appendChild(faceScript);

            // ---- best-effort phone/object detection via coco-ssd ----
            const tfScript = doc.createElement('script');
            tfScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js';
            tfScript.onload = function() {{
                const cocoScript = doc.createElement('script');
                cocoScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.2/dist/coco-ssd.min.js';
                cocoScript.onload = function() {{
                    window.top.cocoSsd.load().then(function(model) {{
                        setInterval(async function() {{
                            if (!videoEl.srcObject || videoEl.readyState < 2) return;
                            let preds;
                            try {{ preds = await model.detect(videoEl); }} catch (e) {{ return; }}
                            const phone = preds.find(function(p) {{ return p.class === 'cell phone' && p.score > 0.55; }});
                            const now = Date.now();
                            const st_ = window.top.__proctorState;
                            if (phone && (now - st_.phoneLastSeen) > PHONE_COOLDOWN_MS) {{
                                st_.phoneLastSeen = now;
                                addEvent('Possible phone detected', 'High',
                                    'An object resembling a mobile phone was seen in the camera frame');
                            }}
                        }}, 2500);
                    }});
                }};
                doc.head.appendChild(cocoScript);
            }};
            doc.head.appendChild(tfScript);
        }})();
        </script>
        """,
        height=0,
    )


def render_fullscreen_consent_prompt():
    """Shown once, at the very start of the interview. Requests fullscreen
    via a real in-frame button click (fullscreen requires a genuine user
    gesture — it can't be triggered automatically from a Streamlit rerun),
    then signals Python via the same query-param navigation trick used
    elsewhere in this file so the prompt doesn't show again."""
    st.components.v1.html(
        """
        <div style="font-family:sans-serif;background:#2d1b1b;border:1px solid #7a3b3b;
                    border-radius:8px;padding:14px;color:#f5d6d6;">
          <strong>⚠️ Proctoring notice</strong>
          <p style="margin:8px 0;">Please remain on the interview screen. Switching tabs or leaving the
          interview window will be recorded as a proctoring event.</p>
          <button id="fsBtn" style="padding:8px 14px;border-radius:6px;border:none;
                  background:#e53e3e;color:white;cursor:pointer;">Enter Fullscreen &amp; Begin</button>
        </div>
        <script>
        document.getElementById('fsBtn').onclick = async function() {
            try {
                await window.top.document.documentElement.requestFullscreen();
                // Keyboard Lock can swallow shortcuts like Alt+Tab/Cmd+Tab
                // while in fullscreen, but only in Chromium browsers over
                // HTTPS, and only if the browser/OS allows it — it is not
                // guaranteed to work, and it never blocks clicking another
                // window or app directly.
                if (navigator.keyboard && navigator.keyboard.lock) {
                    try { await navigator.keyboard.lock(); } catch (e) {
                        console.warn("Keyboard lock unavailable.", e);
                    }
                }
            } catch (e) {
                console.warn("Fullscreen request failed or was denied.", e);
            }
            try {
                window.top.location.search = "?fsack=1";
            } catch (e) {
                console.warn("Could not update the top window's URL; continuing without fullscreen ack.", e);
            }
        };
        </script>
        """,
        height=140,
    )


# ---------- GLOBAL: strict-mode violation handler ----------
# The proctoring script (see render_proctoring_system) navigates the top
# window to "?violation=1&pevents=..." the moment the candidate returns to
# a tab they switched away from, if strict mode is on. This is checked
# before anything else is rendered so the interview ends immediately,
# regardless of which stage the candidate was on when it fired.
if st.query_params.get("violation") == "1" and st.session_state.stage in ("interview", "reacting"):
    raw_events = st.query_params.get("pevents")
    if raw_events:
        try:
            st.session_state.proctor_log = json.loads(raw_events)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    st.session_state.disqualified = True
    st.session_state.disqualify_reason = (
        "Strict mode is on and the interview tab was switched away from or minimized."
    )
    st.session_state.stage = "feedback"
    st.query_params.clear()
    st.rerun()

# ---------- UI: LANDING STAGE ----------
if st.session_state.stage == "landing":
    render_navbar()

    st.markdown(
        f"""
        <div class="ia-hero">
            <h1>Interview<span>AI</span></h1>
            <p class="tagline">{APP_TAGLINE}</p>
            <p class="desc">
                A realistic, AI-powered mock interview platform. Pick a role, answer on camera,
                and get instant, structured feedback on your technical depth, communication, and
                confidence — just like a real interview panel would give you.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("🚀 Start Interview", use_container_width=True, type="primary"):
            st.session_state.stage = "setup"
            st.rerun()

    st.write("")
    st.write("")

    feature_cols = st.columns(4)
    features = [
        ("🧠", "AI-Powered Questions", "Role-aware technical, HR, and behavioral questions generated for your target job."),
        ("🎥", "Video Interview", "Answer on camera through your browser, just like a real virtual interview."),
        ("🛡️", "Real-Time Monitoring", "Lightweight, transparent proctoring flags tab switches and attention shifts."),
        ("📊", "Performance Feedback", "Get scored on technical depth, communication, confidence, and relevance."),
    ]
    for col, (icon, title, desc) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="ia-card">
                    <div class="ia-icon">{icon}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")

    stat_cols = st.columns(4)
    stats = [
        ("11", "Languages Supported"),
        ("4", "Interview Types"),
        ("5–15", "Questions per Session"),
        ("100%", "Browser-Based, No Install"),
    ]
    for col, (num, label) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="ia-stat">
                    <div class="ia-stat-num">{num}</div>
                    <div class="ia-stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------- UI: SETUP STAGE ----------
elif st.session_state.stage == "setup":
    render_navbar("Interview Setup")
    st.markdown("### Set up your mock interview")
    st.caption("Configure the interview to match the role you're preparing for, then review the summary and start.")

    col1, col2 = st.columns(2)
    with col1:
        role_input = st.text_input("🎯 Target role / topic", placeholder="e.g. Python Backend Developer")
        interview_type = st.selectbox("🗂️ Interview type", options=INTERVIEW_TYPES, index=INTERVIEW_TYPES.index(st.session_state.interview_type))
        num_questions = st.select_slider("🔢 Number of questions", options=QUESTION_COUNT_OPTIONS, value=st.session_state.num_questions)
    with col2:
        experience_level = st.selectbox("📈 Experience level", options=EXPERIENCE_LEVELS, index=EXPERIENCE_LEVELS.index(st.session_state.experience_level))
        language_name = st.selectbox("🌐 Language", options=list(LANGUAGES.keys()), index=0)
        difficulty = st.select_slider("⚙️ Difficulty", options=DIFFICULTY_LEVELS, value=st.session_state.difficulty)

    if role_input.strip():
        st.markdown(
            f"""
            <div class="ia-summary">
                <h4>📋 Interview Summary</h4>
                <div class="ia-row"><span>Role / Topic</span><b>{role_input.strip()}</b></div>
                <div class="ia-row"><span>Experience Level</span><b>{experience_level}</b></div>
                <div class="ia-row"><span>Interview Type</span><b>{interview_type}</b></div>
                <div class="ia-row"><span>Number of Questions</span><b>{num_questions}</b></div>
                <div class="ia-row"><span>Language</span><b>{language_name}</b></div>
                <div class="ia-row"><span>Difficulty</span><b>{difficulty}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**🛡️ Proctoring & privacy notice**")
    st.caption(
        "This interview uses your camera and microphone for recording and for automated "
        "proctoring (checking that your face is visible, that you stay on this screen, and "
        "that no unauthorized materials are visible). We can only detect that you left the "
        "interview window, never what site or app you switched to. Face/gaze/phone checks "
        "run in your browser and flag possible issues for review — they don't make final "
        "accusations. No web page can truly stop a browser from opening another tab; "
        "'Strict mode' below is the closest practical substitute — it ends the interview the "
        "moment you switch away, instead of only logging it."
    )
    strict_mode = st.checkbox(
        "🔒 Strict mode — end the interview immediately if I switch tabs or minimize the window "
        "(otherwise, switches are just logged, up to "
        f"{MAX_ALLOWED_SCREEN_SWITCHES} before a warning).",
        value=st.session_state.strict_mode,
    )
    consent = st.checkbox(
        "I consent to camera/microphone recording and automated proctoring for this interview.",
        value=st.session_state.consent_given,
    )
    st.session_state.consent_given = consent

    back_col, start_col = st.columns([1, 3])
    with back_col:
        if st.button("← Back"):
            st.session_state.stage = "landing"
            st.rerun()
    with start_col:
        if st.button("Start Interview →", disabled=not (role_input.strip() and consent), type="primary"):
            st.session_state.role = role_input.strip()
            st.session_state.language = LANGUAGES[language_name]
            st.session_state.experience_level = experience_level
            st.session_state.interview_type = interview_type
            st.session_state.difficulty = difficulty
            st.session_state.num_questions = num_questions
            st.session_state.strict_mode = strict_mode
            st.session_state.disqualified = False
            st.session_state.disqualify_reason = ""
            st.session_state.history = []
            st.session_state.question_queue = pick_interview_questions(
                st.session_state.role, num_questions, interview_type, difficulty
            )
            first_question = (
                st.session_state.question_queue.pop(0)["text"]
                if st.session_state.question_queue
                else generate_question(st.session_state.role, [])
            )
            st.session_state.current_question = first_question
            st.session_state.spoken_question = ""
            st.session_state.translated_question = ""
            st.session_state.audio_bytes = None
            st.session_state.recorded_video = None
            st.session_state.transcribed_answer = ""
            st.session_state.proctor_log = []
            st.session_state.fullscreen_prompt_shown = False
            st.session_state.switch_limit_warned = False
            st.session_state.stage = "interview"
            st.rerun()

# ---------- UI: INTERVIEW STAGE ----------
elif st.session_state.stage == "interview":
    render_navbar(f"{st.session_state.role or 'Interview'} · {st.session_state.interview_type}")
    q_num = len(st.session_state.history) + 1

    # Fullscreen requires a genuine click, so handle its ack before anything else.
    if st.query_params.get("fsack") == "1":
        st.session_state.fullscreen_prompt_shown = True
        st.query_params.clear()

    render_proctoring_system(strict_mode=st.session_state.strict_mode)  # no-ops after first mount

    if not st.session_state.fullscreen_prompt_shown:
        render_fullscreen_consent_prompt()

    # Enforce (softly) the configured screen-switch limit once we know about it.
    switch_events = [e for e in st.session_state.proctor_log if e.get("type") == "Tab switched"]
    if len(switch_events) > MAX_ALLOWED_SCREEN_SWITCHES and not st.session_state.switch_limit_warned:
        st.warning(
            f"You've switched away from the interview screen {len(switch_events)} times, "
            f"more than the allowed {MAX_ALLOWED_SCREEN_SWITCHES}. This has been recorded. "
            "You may continue, but repeated switching will be reflected in your proctoring summary."
        )
        st.session_state.switch_limit_warned = True

    if st.session_state.history:
        scores_so_far = [h["score"] for h in st.session_state.history if "score" in h]
        if scores_so_far:
            avg_score = sum(scores_so_far) / len(scores_so_far)
            st.metric("Running score", f"{avg_score:.1f} / 10")

    st.subheader(f"Question {q_num} of {st.session_state.num_questions}")

    # Only re-translate/regenerate audio when the question actually changes —
    # otherwise every rerun (e.g. while typing in the answer box) would fire
    # a translation API call and re-synthesize speech for no reason.
    if st.session_state.spoken_question != st.session_state.current_question:
        st.session_state.translated_question = translate_text(
            st.session_state.current_question, st.session_state.language
        )
        st.session_state.audio_bytes = text_to_speech(
            st.session_state.translated_question, st.session_state.language
        )
        st.session_state.spoken_question = st.session_state.current_question
        st.session_state.recorded_video = None  # reset capture for the new question
        st.session_state.transcribed_answer = ""

    st.write(st.session_state.translated_question)
    render_avatar_with_speech(st.session_state.audio_bytes, unique_id=f"q{q_num}")

    st.write("Record your answer on video, then upload the clip below:")

    # Real in-browser video recording using plain JavaScript (MediaRecorder API).
    # This needs NO server connection at all — recording happens entirely on
    # your device, so there's no STUN/TURN/NAT issue, and this embedding method
    # (st.components.v1.html) reliably gets camera permission, unlike the
    # custom declare_component version we tried. When you click "Stop", it
    # downloads the clip as a .webm file, which you then upload just below.
    st.components.v1.html(
        """
        <div style="font-family: sans-serif;">
          <video id="preview" autoplay muted playsinline
                 style="width:100%; max-width:480px; border-radius:8px; background:#000;"></video>
          <br><br>
          <button id="startBtn">🔴 Start Recording</button>
          <button id="stopBtn" disabled>⏹ Stop & Download</button>
          <p id="status" style="color:gray;"></p>
        </div>
        <script>
        let mediaRecorder;
        let chunks = [];
        const preview = document.getElementById('preview');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');

        startBtn.onclick = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            preview.srcObject = stream;
            mediaRecorder = new MediaRecorder(stream);
            chunks = [];
            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'answer.webm';
                a.click();
                status.innerText = 'Downloaded! Now upload that file below.';
                stream.getTracks().forEach(track => track.stop());
            };
            mediaRecorder.start();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            status.innerText = 'Recording...';
        };

        stopBtn.onclick = () => {
            mediaRecorder.stop();
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };
        </script>
        """,
        height=420,
    )

    uploaded_clip = st.file_uploader(
        "Upload your recorded answer.webm",
        type=["webm", "mp4"],
        key=f"upload_{q_num}",
    )
    if uploaded_clip is not None and st.session_state.recorded_video != uploaded_clip:
        st.session_state.recorded_video = uploaded_clip
        with st.spinner("Transcribing your answer..."):
            transcript = transcribe_video(uploaded_clip)
        st.session_state.transcribed_answer = transcript
        # Streamlit text_area's `value=` argument is only used the FIRST time
        # a widget with this key is created — once that key exists in
        # session_state, later reruns ignore `value=` and keep whatever's
        # already there. So we seed the key directly here, before the widget
        # below gets created, to make the transcription actually show up.
        st.session_state[f"answer_{q_num}"] = transcript

    if uploaded_clip is not None:
        st.video(uploaded_clip)

    st.caption("Transcribed automatically from your video — review and edit if needed before submitting.")
    answer = st.text_area(
        "Your answer",
        key=f"answer_{q_num}",
    )

    if st.button("Submit Answer", disabled=not answer.strip()):
        with st.spinner("Evaluating your answer..."):
            reaction_data = generate_reaction(
                st.session_state.role, st.session_state.current_question, answer.strip()
            )

        reaction_display = translate_text(reaction_data["reaction"], st.session_state.language)
        st.session_state.history.append({
            "question": st.session_state.current_question,
            "answer": answer.strip(),
            "score": reaction_data["score"],
            "reaction": reaction_display,
        })
        st.session_state.reaction_audio = text_to_speech(reaction_display, st.session_state.language)
        st.session_state.stage = "reacting"
        st.rerun()

# ---------- UI: REACTING STAGE (instant per-answer feedback) ----------
elif st.session_state.stage == "reacting":
    last = st.session_state.history[-1]
    reaction_id = f"r{len(st.session_state.history)}"

    st.subheader(f"Score: {last['score']} / 10")
    st.write(last["reaction"])
    render_avatar_with_speech(
        st.session_state.reaction_audio, unique_id=reaction_id, auto_advance=True
    )

    is_last_question = len(st.session_state.history) >= st.session_state.num_questions
    button_label = "See Final Feedback" if is_last_question else "Next Question"

    def _advance():
        if is_last_question:
            st.session_state.stage = "feedback"
        elif st.session_state.question_queue:
            st.session_state.current_question = st.session_state.question_queue.pop(0)["text"]
            st.session_state.stage = "interview"
        else:
            # Bank exhausted (shouldn't normally happen) — fall back to the
            # original LLM-generated adaptive question so the app still works.
            st.session_state.current_question = generate_question(
                st.session_state.role, st.session_state.history
            )
            st.session_state.stage = "interview"

    def _sync_proctor_log():
        """Pull the client-side proctoring event log (kept on window.top,
        see render_proctoring_system) into session_state so it survives
        into the final report."""
        raw_events = st.query_params.get("pevents")
        if raw_events:
            try:
                st.session_state.proctor_log = json.loads(raw_events)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    # The avatar's JS navigates to "?advance=<reaction_id>&pevents=<json>" the
    # instant the reaction audio finishes — this catches that, syncs the
    # proctoring log, and moves on automatically, with NO click needed. The
    # button below still works as a manual fallback (some browsers/embeds can
    # block the JS navigation trick), though in that case proctoring events
    # only get synced again on the next automatic advance.
    if st.query_params.get("advance") == reaction_id:
        _sync_proctor_log()
        st.query_params.clear()
        _advance()
        st.rerun()

    st.caption("Moving to the next step automatically once the AI finishes speaking...")
    if st.button(button_label):
        # Bug fix: this manual fallback previously skipped _sync_proctor_log(),
        # so proctoring events for this stretch were silently dropped whenever
        # someone clicked through instead of waiting for auto-advance. It's
        # still a no-op if the browser hasn't navigated with ?pevents= yet,
        # but calling it keeps both paths consistent.
        _sync_proctor_log()
        _advance()
        st.rerun()

# ---------- UI: FEEDBACK STAGE ----------
elif st.session_state.stage == "feedback":
    render_navbar("Interview Report")
    st.title("📋 Interview Feedback")

    if st.session_state.disqualified:
        st.error(
            "⛔ Interview ended early — " + st.session_state.disqualify_reason +
            f" You answered {len(st.session_state.history)} question(s) before this happened; "
            "the report below only covers those."
        )

    if not st.session_state.history:
        st.info("No answers were recorded, so there's no report to generate.")
        if st.button("🏠 Back to Home"):
            st.session_state.stage = "landing"
            st.session_state.disqualified = False
            st.session_state.disqualify_reason = ""
            st.rerun()
        st.stop()

    with st.spinner("Generating your interview report..."):
        report = generate_final_report(st.session_state.role, st.session_state.history)
        feedback = generate_feedback(st.session_state.role, st.session_state.history)
        feedback_display = translate_text(feedback, st.session_state.language)

    st.subheader("Interview Score")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Technical", f"{report['technical_score']}/10")
    c2.metric("Communication", f"{report['communication_score']}/10")
    c3.metric("Confidence", f"{report['confidence_score']}/10")
    c4.metric("Relevance", f"{report['relevance_score']}/10")
    c5.metric("Overall", f"{report['overall_score']}/10")

    if report["strengths"]:
        st.markdown("**Strengths**")
        for s in report["strengths"]:
            st.markdown(f"- {s}")
    if report["improvements"]:
        st.markdown("**Areas for improvement**")
        for s in report["improvements"]:
            st.markdown(f"- {s}")

    st.write(feedback_display)

    st.divider()
    st.subheader("🛡️ Proctoring Summary")
    st.caption(
        "This reflects automated, browser-based monitoring during the interview. Events are "
        "possible-issue flags for human review, not confirmed findings of cheating."
    )
    log = st.session_state.proctor_log
    if not log:
        st.write("No proctoring events were recorded (or camera/monitoring permissions were not granted).")
    else:
        high = sum(1 for e in log if e.get("severity") == "High")
        medium = sum(1 for e in log if e.get("severity") == "Medium")
        low = sum(1 for e in log if e.get("severity") == "Low")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Total events", len(log))
        pc2.metric("High severity", high)
        pc3.metric("Medium severity", medium)
        pc4.metric("Low severity", low)

        with st.expander("View full proctoring event log", expanded=False):
            for e in sorted(log, key=lambda x: x.get("elapsed_seconds", 0)):
                secs = e.get("elapsed_seconds", 0)
                mm, ss = divmod(int(secs), 60)
                st.write(f"`{mm:02d}:{ss:02d}` — {e.get('description', e.get('type', ''))} — **{e.get('severity', '')}**")

    st.divider()
    st.subheader("Transcript")
    for i, h in enumerate(st.session_state.history, start=1):
        score_suffix = f" — Score: {h['score']}/10" if "score" in h else ""
        st.markdown(f"**Q{i}: {h['question']}**{score_suffix}")
        st.write(h["answer"])
        if "reaction" in h:
            st.caption(f"Interviewer reaction: {h['reaction']}")

    restart_col, home_col = st.columns(2)

    def _reset_for_new_session():
        st.session_state.history = []
        st.session_state.current_question = ""
        st.session_state.spoken_question = ""
        st.session_state.translated_question = ""
        st.session_state.audio_bytes = None
        st.session_state.recorded_video = None
        st.session_state.reaction_audio = None
        st.session_state.language = "en"
        st.session_state.question_queue = []
        st.session_state.proctor_log = []
        st.session_state.fullscreen_prompt_shown = False
        st.session_state.switch_limit_warned = False
        st.session_state.disqualified = False
        st.session_state.disqualify_reason = ""
        # Note: consent_given is intentionally NOT reset — re-consent isn't
        # required to start a new session in the same browser tab. Also, the
        # window.top proctoring overlay from the previous interview keeps
        # running (see render_proctoring_system's guard) until the page is
        # reloaded; that's a known limitation of the no-custom-component
        # approach this app uses.

    with restart_col:
        if st.button("🔁 Start New Interview", type="primary", use_container_width=True):
            _reset_for_new_session()
            st.session_state.stage = "setup"
            st.rerun()
    with home_col:
        if st.button("🏠 Back to Home", use_container_width=True):
            _reset_for_new_session()
            st.session_state.stage = "landing"
            st.rerun()
