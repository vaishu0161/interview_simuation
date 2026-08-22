"""
AI Interview Simulator — Base Prototype (Step 1-3)
----------------------------------------------------
This is the foundation: text-based Q&A loop using Groq.
Video, voice (gTTS), and Whisper transcription get added on top of this
in later steps — this file proves the core AI question-generation
logic works before anything else is layered in.

Setup:
    pip install streamlit groq

Run:
    streamlit run interview_simulator_app.py

You'll need a free Groq API key from https://console.groq.com
Set it as an environment variable before running:
    export GROQ_API_KEY="your_key_here"      (Mac/Linux)
    setx GROQ_API_KEY "your_key_here"         (Windows)
"""

import os
import streamlit as st
from groq import Groq

# ---------- CONFIG ----------
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast + good quality on Groq's free tier
MAX_QUESTIONS = 5  # fixed number of questions per session (keeps it predictable)

# ---------- SETUP ----------
st.set_page_config(page_title="AI Interview Simulator", page_icon="🎤")

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Set it as an environment variable before running.")
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


# ---------- UI: SETUP STAGE ----------
if st.session_state.stage == "setup":
    st.title("🎤 AI Interview Simulator")
    st.write("Enter your target role or topic to begin a simulated interview.")

    role_input = st.text_input("Target role / topic", placeholder="e.g. Python Backend Developer")

    if st.button("Start Interview", disabled=not role_input.strip()):
        st.session_state.role = role_input.strip()
        st.session_state.history = []
        st.session_state.current_question = generate_question(st.session_state.role, [])
        st.session_state.stage = "interview"
        st.rerun()

# ---------- UI: INTERVIEW STAGE ----------
elif st.session_state.stage == "interview":
    q_num = len(st.session_state.history) + 1
    st.subheader(f"Question {q_num} of {MAX_QUESTIONS}")
    st.write(st.session_state.current_question)

    answer = st.text_area("Your answer", key=f"answer_{q_num}")

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
        st.rerun()