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
    pip install streamlit groq gTTS

Run:
    streamlit run main2.py

You'll need a free Groq API key from https://console.groq.com
Set it as an environment variable before running (or as a Streamlit secret):
    export GROQ_API_KEY="your_key_here"      (Mac/Linux)
    setx GROQ_API_KEY "your_key_here"         (Windows)
"""

import base64
import io
import json
import os
import tempfile
import streamlit as st
import whisper
from groq import Groq
from gtts import gTTS

# ---------- CONFIG ----------
GROQ_MODEL = "openai/gpt-oss-20b"  # fast + good quality on Groq's free tier
MAX_QUESTIONS = 1  # fixed number of questions per session (keeps it predictable)
WHISPER_MODEL_SIZE = "base"  # small + fast enough for CPU, decent accuracy


@st.cache_resource
def load_whisper_model():
    """Loaded once and cached across reruns/users — downloading the model
    weights every rerun would be far too slow."""
    return whisper.load_model(WHISPER_MODEL_SIZE)

# ---------- SETUP ----------
st.set_page_config(page_title="AI Interview Simulator", page_icon="🎤")

api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Add GROQ_API_KEY to Streamlit Secrets or set it as an environment variable.")
    st.stop()

client = Groq(api_key=api_key)

# ---------- SESSION STATE ----------
# session_state persists data across reruns (Streamlit reruns the whole
# script on every interaction, so this is where we keep the conversation).
if "stage" not in st.session_state:
    st.session_state.stage = "setup"          # setup -> interview -> feedback
if "role" not in st.session_state:
    st.session_state.role = ""
if "history" not in st.session_state:
    st.session_state.history = []             # list of {"question": ..., "answer": ...}
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "spoken_question" not in st.session_state:
    st.session_state.spoken_question = ""      # tracks which question we've already voiced
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "recorded_video" not in st.session_state:
    st.session_state.recorded_video = None  # holds the uploaded video clip for the current question
if "transcribed_answer" not in st.session_state:
    st.session_state.transcribed_answer = ""  # Whisper's transcription of the current answer
if "reaction_audio" not in st.session_state:
    st.session_state.reaction_audio = None  # spoken version of the last reaction


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


# ---------- VOICE HELPER ----------
def text_to_speech(text: str) -> bytes:
    """Convert text into spoken audio (in-memory, no temp files needed)."""
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
    auto_advance_js = (
        f'window.top.location.search = "?advance={unique_id}";'
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


# ---------- UI: SETUP STAGE ----------
if st.session_state.stage == "setup":
    st.title("🎤 AI Interview Simulator")
    st.write("Enter your target role or topic to begin a simulated interview.")

    role_input = st.text_input("Target role / topic", placeholder="e.g. Python Backend Developer")

    if st.button("Start Interview", disabled=not role_input.strip()):
        st.session_state.role = role_input.strip()
        st.session_state.history = []
        st.session_state.current_question = generate_question(st.session_state.role, [])
        st.session_state.spoken_question = ""
        st.session_state.audio_bytes = None
        st.session_state.recorded_video = None
        st.session_state.transcribed_answer = ""
        st.session_state.stage = "interview"
        st.rerun()

# ---------- UI: INTERVIEW STAGE ----------
elif st.session_state.stage == "interview":
    q_num = len(st.session_state.history) + 1

    if st.session_state.history:
        scores_so_far = [h["score"] for h in st.session_state.history if "score" in h]
        if scores_so_far:
            avg_score = sum(scores_so_far) / len(scores_so_far)
            st.metric("Running score", f"{avg_score:.1f} / 10")

    st.subheader(f"Question {q_num} of {MAX_QUESTIONS}")
    st.write(st.session_state.current_question)

    # Only regenerate audio when the question actually changes —
    # otherwise it would re-speak the same question on every rerun.
    if st.session_state.spoken_question != st.session_state.current_question:
        st.session_state.audio_bytes = text_to_speech(st.session_state.current_question)
        st.session_state.spoken_question = st.session_state.current_question
        st.session_state.recorded_video = None  # reset capture for the new question
        st.session_state.transcribed_answer = ""

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

        st.session_state.history.append({
            "question": st.session_state.current_question,
            "answer": answer.strip(),
            "score": reaction_data["score"],
            "reaction": reaction_data["reaction"],
        })
        st.session_state.reaction_audio = text_to_speech(reaction_data["reaction"])
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

    is_last_question = len(st.session_state.history) >= MAX_QUESTIONS
    button_label = "See Final Feedback" if is_last_question else "Next Question"

    def _advance():
        if is_last_question:
            st.session_state.stage = "feedback"
        else:
            st.session_state.current_question = generate_question(
                st.session_state.role, st.session_state.history
            )
            st.session_state.stage = "interview"

    # The avatar's JS navigates to "?advance=<reaction_id>" the instant the
    # reaction audio finishes — this catches that and moves on automatically,
    # with NO click needed. The button below still works as a manual
    # fallback (some browsers/embeds can block the JS navigation trick).
    if st.query_params.get("advance") == reaction_id:
        st.query_params.clear()
        _advance()
        st.rerun()

    st.caption("Moving to the next step automatically once the AI finishes speaking...")
    if st.button(button_label):
        _advance()
        st.rerun()

# ---------- UI: FEEDBACK STAGE ----------
elif st.session_state.stage == "feedback":
    st.title("📋 Interview Feedback")

    scores = [h["score"] for h in st.session_state.history if "score" in h]
    if scores:
        st.metric("Overall Score", f"{sum(scores) / len(scores):.1f} / 10")

    with st.spinner("Generating feedback..."):
        feedback = generate_feedback(st.session_state.role, st.session_state.history)

    st.write(feedback)

    st.divider()
    st.subheader("Transcript")
    for i, h in enumerate(st.session_state.history, start=1):
        score_suffix = f" — Score: {h['score']}/10" if "score" in h else ""
        st.markdown(f"**Q{i}: {h['question']}**{score_suffix}")
        st.write(h["answer"])
        if "reaction" in h:
            st.caption(f"Interviewer reaction: {h['reaction']}")

    if st.button("Start New Interview"):
        st.session_state.stage = "setup"
        st.session_state.history = []
        st.session_state.current_question = ""
        st.session_state.spoken_question = ""
        st.session_state.audio_bytes = None
        st.session_state.recorded_video = None
        st.session_state.reaction_audio = None
        st.rerun()