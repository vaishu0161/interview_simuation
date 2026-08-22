"""
AI Interview Simulator — Prototype (Step 1-5, revised)
----------------------------------------------------
Text-based Q&A loop using Groq + gTTS voice, with a per-answer
video/audio recording step (st.camera_input) instead of live
WebRTC streaming — live streaming via streamlit-webrtc kept
timing out on Streamlit Community Cloud due to NAT/STUN-TURN
issues, so this switches to record-then-capture per answer,
which needs no peer connection at all.

Setup:
    pip install streamlit groq gTTS

Run:
    streamlit run mainapp.py

You'll need a free Groq API key from https://console.groq.com
Set it as an environment variable before running (or as a Streamlit secret):
    export GROQ_API_KEY="your_key_here"      (Mac/Linux)
    setx GROQ_API_KEY "your_key_here"         (Windows)
"""

import io
import os
import streamlit as st
from groq import Groq
from gtts import gTTS

# ---------- CONFIG ----------
GROQ_MODEL = "openai/gpt-oss-20b"  # fast + good quality on Groq's free tier
MAX_QUESTIONS = 5  # fixed number of questions per session (keeps it predictable)

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
    st.session_state.recorded_video = None  # holds the captured video/photo for the current question


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


# ---------- VOICE HELPER ----------
def text_to_speech(text: str) -> bytes:
    """Convert text into spoken audio (in-memory, no temp files needed)."""
    tts = gTTS(text=text, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


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
        st.session_state.stage = "interview"
        st.rerun()

# ---------- UI: INTERVIEW STAGE ----------
elif st.session_state.stage == "interview":
    q_num = len(st.session_state.history) + 1
    st.subheader(f"Question {q_num} of {MAX_QUESTIONS}")
    st.write(st.session_state.current_question)

    # Only regenerate audio when the question actually changes —
    # otherwise it would re-speak the same question on every rerun.
    if st.session_state.spoken_question != st.session_state.current_question:
        st.session_state.audio_bytes = text_to_speech(st.session_state.current_question)
        st.session_state.spoken_question = st.session_state.current_question
        st.session_state.recorded_video = None  # reset capture for the new question

    st.audio(st.session_state.audio_bytes, format="audio/mp3", autoplay=True)

    st.write("Look at the camera and answer, then capture a snapshot when you're done:")
    captured = st.camera_input("Record your answer", key=f"camera_{q_num}")
    if captured is not None:
        st.session_state.recorded_video = captured

    st.caption("Speech-to-text transcription of your spoken answer will be wired in next — for now, type your answer below.")
    answer = st.text_area("Your answer (temporary text input until Whisper is wired in)", key=f"answer_{q_num}")

    if st.button("Submit Answer", disabled=not answer.strip()):
        st.session_state.history.append({
            "question": st.session_state.current_question,
            "answer": answer.strip(),
        })

        if len(st.session_state.history) >= MAX_QUESTIONS:
            st.session_state.stage = "feedback"
        else:
            st.session_state.current_question = generate_question(
                st.session_state.role, st.session_state.history
            )
        st.rerun()

# ---------- UI: FEEDBACK STAGE ----------
elif st.session_state.stage == "feedback":
    st.title("📋 Interview Feedback")

    with st.spinner("Generating feedback..."):
        feedback = generate_feedback(st.session_state.role, st.session_state.history)

    st.write(feedback)

    st.divider()
    st.subheader("Transcript")
    for i, h in enumerate(st.session_state.history, start=1):
        st.markdown(f"**Q{i}: {h['question']}**")
        st.write(h["answer"])

    if st.button("Start New Interview"):
        st.session_state.stage = "setup"
        st.session_state.history = []
        st.session_state.current_question = ""
        st.session_state.spoken_question = ""
        st.session_state.audio_bytes = None
        st.session_state.recorded_video = None
        st.rerun()