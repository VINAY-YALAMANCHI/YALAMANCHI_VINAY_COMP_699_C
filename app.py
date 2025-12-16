"""
VinSol-AI: AI-Assisted Interview Simulator
Author: Vinay Yalamanchi
Course: COMP-699 - Professional Seminar
Professor: Dr. David Pitts
Date: December 16, 2025
Company: VinSol-AI Technologies

A comprehensive voice-driven mock interview platform that provides real-time
AI-powered feedback on candidate responses. The system evaluates semantic
relevance, delivery clarity, confidence, and overall performance using
natural language processing and speech recognition technologies.
"""

import streamlit as st
import random
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr

# ==============================
# FILE PATH CONSTANTS
# ==============================

DATA_FILE = "users_data.json"          # Persistent storage for user accounts and interview history
CONFIG_FILE = "app_config.json"        # Application-wide configuration settings
QUESTIONS_FILE = "questions_bank.json" # Role-specific question bank

# ==============================
# DEFAULT CONFIGURATION
# ==============================

DEFAULT_CONFIG = {
    "app_title": "VinSol-AI",
    "app_tagline": "Real Voice - Real-Time AI Feedback - Real Growth",
    "max_questions_per_interview": 4,
    "minimum_answer_words_for_analysis": 60,
    "recommended_answer_word_range": [90, 200],
    "filler_words": [
        "um", "uh", "like", "you know", "so", "well", "basically",
        "literally", "sort of", "kind of", "right", "okay", "actually",
        "honestly", "essentially", "pretty much", "I mean"
    ],
    "pause_indicators": ["...", "--", "——", "…"],
    "example_keywords": [
        "example", "case", "project", "worked on", "built", "created",
        "implemented", "developed", "designed", "led", "managed"
    ],
    "star_method_keywords": [
        "situation", "task", "action", "result", "challenge", "goal",
        "achieved", "impact", "outcome", "delivered", "responsibility",
        "objective"
    ],
    "technical_keywords": [
        "api", "algorithm", "database", "system", "architecture",
        "performance", "debug", "deploy", "scale", "cache", "index",
        "query", "framework", "pattern", "microservice", "cloud",
        "container", "orchestration", "pipeline", "testing", "refactor"
    ],
    "relevance_weight": 0.50,
    "confidence_weight": 0.25,
    "clarity_weight": 0.25
}

# ==============================
# DEFAULT QUESTION BANK
# ==============================

DEFAULT_QUESTIONS = {
    "Software Engineer": [
        "Tell me about yourself and your background in software development.",
        "Describe a challenging technical problem you solved recently.",
        "Explain Object-Oriented Programming principles with examples.",
        "How do you ensure code quality in your projects?",
        "Walk me through how you would design a scalable web application.",
        "What is your experience with version control systems like Git?",
        "Explain the difference between REST and GraphQL APIs.",
        "How do you approach debugging a complex production issue?",
        "Describe your experience with cloud platforms and services."
    ],
    "Data Scientist": [
        "Tell me about a machine learning project you worked on.",
        "How do you handle missing or noisy data?",
        "Explain overfitting and how to prevent it.",
        "What evaluation metrics do you use for classification vs regression?",
        "Describe the bias-variance tradeoff in detail.",
        "How would you explain a complex model to a non-technical stakeholder?",
        "What is your experience with deep learning frameworks?",
        "How do you approach feature engineering?",
        "Describe a time when your model failed in production and how you handled it."
    ],
    "Product Manager": [
        "How do you prioritize features in a product roadmap?",
        "Describe a product you successfully launched from idea to market.",
        "How do you work cross-functionally with engineering and design?",
        "What key metrics define success for your product?",
        "How do you handle conflicting feedback from stakeholders?",
        "Explain how you conduct user research and validation.",
        "What frameworks do you use for product strategy?",
        "How do you measure customer satisfaction?",
        "Describe your approach to writing product requirements."
    ],
    "UX Designer": [
        "Walk me through your end-to-end design process.",
        "How do you conduct effective user interviews?",
        "What is your experience building and maintaining design systems?",
        "How do you handle conflicting feedback from stakeholders?",
        "Describe how you approach accessibility in your designs.",
        "How do you measure the success of a design change?",
        "Explain the difference between UI and UX with examples.",
        "How do you incorporate user feedback into iterations?",
        "Describe your experience with prototyping tools."
    ]
}

# ==============================
# FILE I/O UTILITIES
# ==============================

def load_json_file(filepath: str, default_content: Any) -> Any:
    """
    Load JSON data from a file. If the file does not exist or is corrupted,
    create it with the provided default content and return the default.
    """
    if not os.path.exists(filepath):
        save_json_file(filepath, default_content)
        return default_content
    
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, IOError) as error:
        st.warning(f"Error loading {filepath}: {str(error)}. Resetting to default.")
        save_json_file(filepath, default_content)
        return default_content

def save_json_file(filepath: str, data: Any) -> None:
    """
    Save data to a JSON file with proper indentation and encoding.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except IOError as error:
        st.error(f"Failed to save {filepath}: {str(error)}")

def get_application_config() -> Dict[str, Any]:
    """Retrieve the application configuration."""
    return load_json_file(CONFIG_FILE, DEFAULT_CONFIG)

def get_question_bank() -> Dict[str, List[str]]:
    """Retrieve the role-based question bank."""
    return load_json_file(QUESTIONS_FILE, DEFAULT_QUESTIONS)

def load_user_database() -> Dict[str, Any]:
    """Load the complete user database."""
    return load_json_file(DATA_FILE, {})

def persist_user_database(users: Dict[str, Any]) -> None:
    """Save the updated user database to disk."""
    save_json_file(DATA_FILE, users)

def format_timestamp() -> str:
    """Generate a formatted timestamp string for logging and records."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==============================
# MODEL LOADING
# ==============================

@st.cache_resource
def initialize_models() -> Tuple[SentenceTransformer, sr.Recognizer]:
    """
    Load the sentence transformer model for semantic analysis and the speech recognizer.
    This function is cached to prevent reloading on every interaction.
    """
    with st.spinner("Initializing AI models (this may take a moment on first launch)..."):
        transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        recognizer = sr.Recognizer()
    return transformer_model, recognizer

sentence_transformer, speech_recognizer = initialize_models()

# ==============================
# PAGE CONFIGURATION AND STYLING
# ==============================

st.set_page_config(
    page_title="VinSol-AI",
    page_icon="microphone",
    layout="centered",
    initial_sidebar_state="collapsed"
)

application_config = get_application_config()

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
    
    .main {{
        background: linear-gradient(135deg, #0a001f 0%, #1a0033 40%, #2d1b69 100%);
        font-family: 'Inter', sans-serif;
        padding: 0;
        margin: 0;
        min-height: 100vh;
        color: #e0e7ff;
    }}
    
    .title {{
        font-size: 11rem;
        font-weight: 900;
        background: linear-gradient(90deg, #c084fc, #f472b6, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -8px;
        margin: 120px 0 20px 0;
        line-height: 0.9;
    }}
    
    .subtitle {{
        font-size: 2.8rem;
        color: #e0e7ff;
        text-align: center;
        font-weight: 300;
        letter-spacing: 3px;
        margin-bottom: 100px;
    }}
    
    .question-card {{
        background: rgba(139,92,246,0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(139,92,246,0.35);
        border-radius: 40px;
        padding: 80px 60px;
        font-size: 3rem;
        line-height: 1.5;
        color: #f0f9ff;
        text-align: center;
        max-width: 1100px;
        margin: 60px auto;
        box-shadow: 0 30px 100px rgba(139,92,246,0.3);
    }}
    
    .answer-box {{
        background: rgba(15,25,45,0.75);
        border-left: 8px solid #22d3ee;
        border-radius: 24px;
        padding: 40px;
        font-size: 1.8rem;
        color: #e0f2fe;
        margin: 50px auto;
        max-width: 1100px;
        line-height: 1.8;
    }}
    
    .feedback-card {{
        background: linear-gradient(90deg, rgba(34,197,94,0.22), rgba(22,163,74,0.1));
        border-left: 10px solid #22d3ee;
        border-radius: 24px;
        padding: 50px;
        font-size: 2.1rem;
        color: #ccfbf1;
        max-width: 1100px;
        margin: 70px auto;
        box-shadow: 0 25px 80px rgba(0,0,0,0.45);
    }}
    
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 40px;
        max-width: 1100px;
        margin: 80px auto;
    }}
    
    .metric-item {{
        background: rgba(255,255,255,0.08);
        padding: 45px 20px;
        border-radius: 32px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }}
    
    .metric-num {{
        font-size: 6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .metric-name {{
        font-size: 1.7rem;
        color: #94a3b8;
        margin-top: 15px;
    }}
    
    .final-score {{
        font-size: 10rem;
        font-weight: 900;
        background: linear-gradient(90deg, #22d3ee, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 100px 0;
    }}
    
    .progress-container {{
        height: 14px;
        background: rgba(255,255,255,0.12);
        border-radius: 7px;
        overflow: hidden;
        max-width: 900px;
        margin: 60px auto;
    }}
    
    .progress-bar {{
        height: 100%;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        border-radius: 7px;
        transition: width 0.8s ease;
    }}
    
    .welcome-header {{
        text-align: center;
        font-size: 3.8rem;
        color: #e0e7ff;
        margin: 60px 0;
    }}
    
    .role-header {{
        text-align: center;
        font-size: 5.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 60px;
    }}
    
    .stButton>button {{
        background: linear-gradient(90deg, #8b5cf6, #ec4899) !important;
        color: white !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        padding: 26px 110px !important;
        border-radius: 60px !important;
        border: none !important;
        box-shadow: 0 30px 80px rgba(139,92,246,0.7) !important;
        margin: 50px auto !important;
        display: block !important;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-18px) !important;
        box-shadow: 0 60px 120px rgba(139,92,246,0.9) !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================
# ANALYSIS ENGINE COMPONENTS
# ==============================

def count_filler_words(text: str, fillers: List[str]) -> int:
    """Count occurrences of filler words in the provided text (case-insensitive)."""
    lowered = text.lower()
    return sum(lowered.count(word.lower()) for word in fillers)

def count_pause_indicators(text: str, markers: List[str]) -> int:
    """Count occurrences of pause markers in transcribed text."""
    return sum(text.count(marker) for marker in markers)

def detect_example_usage(text: str, keywords: List[str]) -> bool:
    """Determine if the response includes concrete examples."""
    pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    return bool(pattern.search(text))

def detect_star_structure(text: str, keywords: List[str]) -> bool:
    """Check if response likely follows STAR method (at least 3 keywords)."""
    lowered = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in lowered)
    return matches >= 3

def count_technical_vocabulary(text: str, keywords: List[str]) -> int:
    """Count domain-specific technical terms in the response."""
    lowered = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lowered)

def calculate_semantic_relevance(question: str, answer: str) -> int:
    """Compute cosine similarity between question and answer embeddings."""
    question_embedding = sentence_transformer.encode(question, convert_to_tensor=True)
    answer_embedding = sentence_transformer.encode(answer, convert_to_tensor=True)
    similarity_score = util.cos_sim(question_embedding, answer_embedding)[0][0].item()
    return max(0, min(100, int(similarity_score * 100)))

def evaluate_clarity_and_confidence(answer: str, config: Dict) -> Tuple[int, int]:
    """
    Calculate clarity and confidence scores based on fluency, length,
    and structural indicators.
    """
    word_count = len(answer.split())
    filler_count = count_filler_words(answer, config["filler_words"])
    pause_count = count_pause_indicators(answer, config["pause_indicators"])

    # Clarity calculation
    clarity_base = 90
    clarity_penalty = (filler_count * 5) + (pause_count * 7)
    if word_count < 70:
        clarity_penalty += 20
    clarity_score = max(15, min(98, clarity_base - clarity_penalty))

    # Confidence calculation
    confidence_base = 55
    length_bonus = 35 if word_count > 140 else max(0, (word_count - 60) * 0.6)
    example_bonus = 20 if detect_example_usage(answer, config["example_keywords"]) else 0
    filler_penalty = filler_count * 3
    confidence_score = confidence_base + length_bonus + example_bonus - filler_penalty
    confidence_score = max(20, min(98, int(confidence_score)))

    return clarity_score, confidence_score

def generate_feedback_phrases(metrics: Dict, config: Dict) -> List[str]:
    """Generate a curated list of feedback phrases based on performance metrics."""
    phrases = []

    relevance = metrics["relevance"]
    if relevance >= 95:
        phrases.append(random.choice([
            "Exceptional relevance - perfectly aligned with the question.",
            "Outstanding understanding of the core topic."
        ]))
    elif relevance >= 88:
        phrases.append("Strong relevance with excellent focus on key points.")
    elif relevance >= 80:
        phrases.append("Good relevance and clear connection to the question.")
    elif relevance >= 65:
        phrases.append("Moderate relevance - mostly on track with room for tighter focus.")
    else:
        phrases.append("Limited relevance - consider addressing the question more directly.")

    if detect_star_structure(metrics["answer"], config["star_method_keywords"]):
        phrases.append("Effective use of structured response framework (STAR method).")

    if detect_example_usage(metrics["answer"], config["example_keywords"]):
        word_count = len(metrics["answer"].split())
        if word_count > 120:
            phrases.append("Strong incorporation of detailed real-world examples.")
        else:
            phrases.append("Appropriate use of examples to support points.")

    word_count = len(metrics["answer"].split())
    if word_count >= 180:
        phrases.append("Excellent depth and comprehensive coverage.")
    elif word_count >= 130:
        phrases.append("Solid depth with good level of detail.")
    elif word_count >= 90:
        phrases.append("Adequate content - consider expanding with examples.")
    else:
        phrases.append(f"Response length: {word_count} words - aim for more elaboration.")

    fillers = count_filler_words(metrics["answer"], config["filler_words"])
    if fillers == 0:
        phrases.append("Excellent fluency with no filler words.")
    elif fillers <= 2:
        phrases.append(f"High fluency with minimal fillers ({fillers}).")
    elif fillers <= 6:
        phrases.append(f"Moderate filler word usage ({fillers}) - practice confident pauses.")
    else:
        phrases.append(f"Significant filler usage ({fillers}) - focus on reducing for stronger delivery.")

    score = metrics["score"]
    if score >= 92:
        phrases.append("Outstanding overall performance.")
    elif score >= 85:
        phrases.append("Strong performance suitable for advanced rounds.")
    elif score >= 75:
        phrases.append("Solid performance with clear potential.")

    random.shuffle(phrases)
    return phrases[:6]

def perform_comprehensive_analysis(question: str, answer: str, config: Dict) -> Dict[str, Any]:
    """
    Execute full analysis pipeline on a candidate response.
    Returns structured metrics and detailed feedback.
    """
    cleaned_answer = answer.strip()
    
    if len(cleaned_answer) < 30:
        return {
            "relevance": 5,
            "confidence": 15,
            "clarity": 20,
            "score": 13,
            "feedback": "Response too brief - please provide a detailed answer (at least 45 seconds of speech)."
        }

    relevance_score = calculate_semantic_relevance(question, cleaned_answer)
    clarity_score, confidence_score = evaluate_clarity_and_confidence(cleaned_answer, config)

    overall_score = int(
        config["relevance_weight"] * relevance_score +
        config["confidence_weight"] * confidence_score +
        config["clarity_weight"] * clarity_score
    )

    feedback_phrases = generate_feedback_phrases({
        "relevance": relevance_score,
        "answer": cleaned_answer,
        "score": overall_score
    }, config)

    feedback_text = " • ".join(feedback_phrases)

    return {
        "relevance": relevance_score,
        "confidence": confidence_score,
        "clarity": clarity_score,
        "score": overall_score,
        "feedback": feedback_text
    }

# ==============================
# SPEECH AND AUDIO HANDLING
# ==============================

def deliver_question_audio(text: str) -> None:
    """Use browser text-to-speech to read the question aloud."""
    escaped_text = text.replace('"', '\\"')
    javascript = f'<script>speechSynthesis.speak(new SpeechSynthesisUtterance("{escaped_text}"))</script>'
    st.components.v1.html(javascript, height=0)

def capture_and_transcribe() -> str:
    """Capture audio from microphone and transcribe using Google Speech Recognition."""
    try:
        with sr.Microphone() as source:
            speech_recognizer.adjust_for_ambient_noise(source, duration=1.0)
            st.info("Recording in progress. Speak clearly. Silence for 6 seconds will end recording.")
            audio_data = speech_recognizer.listen(source, phrase_time_limit=None, timeout=5)
        
        with st.spinner("Processing speech to text..."):
            transcribed_text = speech_recognizer.recognize_google(audio_data)
            return transcribed_text if transcribed_text.strip() else "[No content detected]"
    
    except sr.WaitTimeoutError:
        return "[No speech detected within timeout period]"
    except sr.UnknownValueError:
        return "[Audio unclear - unable to transcribe]"
    except sr.RequestError as error:
        return f"[Transcription service error: {str(error)}]"
    except Exception as error:
        return f"[Unexpected error: {str(error)}]"

# ==============================
# SESSION STATE MANAGEMENT
# ==============================

if "stage" not in st.session_state:
    st.session_state.stage = "auth"

if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ==============================
# AUTHENTICATION INTERFACE
# ==============================

def display_authentication_page():
    """Render login and registration interface."""
    st.markdown(f"<div class='title'>{application_config['app_title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{application_config['app_tagline']}</div>", unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            email_input = st.text_input("Email Address")
            password_input = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")
            
            if login_submitted:
                users = load_user_database()
                if email_input in users and users[email_input]["password"] == password_input:
                    st.session_state.current_user = email_input
                    saved_interview = users[email_input].get("interview_data")
                    if saved_interview:
                        for key, value in saved_interview.items():
                            st.session_state[key] = value
                        st.session_state.stage = "results"
                    else:
                        st.session_state.stage = "welcome"
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    with register_tab:
        with st.form("register_form"):
            full_name = st.text_input("Full Name")
            email_reg = st.text_input("Email Address")
            password_reg = st.text_input("Password", type="password")
            register_submitted = st.form_submit_button("Create Account")
            
            if register_submitted:
                if not all([full_name.strip(), email_reg.strip(), password_reg]):
                    st.error("All fields are required.")
                else:
                    users = load_user_database()
                    if email_reg in users:
                        st.error("This email is already registered.")
                    else:
                        users[email_reg] = {
                            "name": full_name.strip(),
                            "password": password_reg,
                            "interview_data": None,
                            "registered_at": format_timestamp()
                        }
                        persist_user_database(users)
                        st.success("Account created successfully. Please log in.")

# ==============================
# WELCOME AND ROLE SELECTION
# ==============================

def display_welcome_page():
    """Render page for name entry and role selection."""
    st.markdown(f"<div class='title'>{application_config['app_title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{application_config['app_tagline']}</div>", unsafe_allow_html=True)

    users = load_user_database()
    default_name = users.get(st.session_state.current_user, {}).get("name", "")

    candidate_name = st.text_input(
        "Full Name",
        placeholder="Enter your full name",
        value=default_name,
        label_visibility="collapsed"
    )

    question_bank = get_question_bank()
    selected_role = st.selectbox(
        "Select Interview Role",
        options=list(question_bank.keys()),
        label_visibility="collapsed"
    )

    if st.button("Start Interview Session"):
        if not candidate_name.strip():
            st.error("Name is required.")
        else:
            questions = random.sample(
                question_bank[selected_role],
                application_config["max_questions_per_interview"]
            )
            st.session_state.update({
                "user": candidate_name.strip(),
                "role": selected_role,
                "questions": questions,
                "current_q": 0,
                "responses": [],
                "last_answer": None,
                "stage": "interview"
            })
            st.rerun()

# ==============================
# INTERVIEW SESSION INTERFACE
# ==============================

def display_interview_session():
    """Render the active interview question and response collection interface."""
    st.markdown(f"<div class='welcome-header'>Welcome, {st.session_state.user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='role-header'>{st.session_state.role} Interview</div>", unsafe_allow_html=True)

    progress_ratio = st.session_state.current_q / application_config["max_questions_per_interview"]
    st.markdown(
        f"<div class='progress-container'><div class='progress-bar' style='width:{progress_ratio * 100}%'></div></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align:center;color:#94a3b8;font-size:1.9rem;margin:40px;'>"
        f"Question {st.session_state.current_q + 1} of {application_config['max_questions_per_interview']}</div>",
        unsafe_allow_html=True
    )

    current_question = st.session_state.questions[st.session_state.current_q]
    st.markdown(f"<div class='question-card'>{current_question}</div>", unsafe_allow_html=True)
    
    deliver_question_audio(current_question)

    record_column_left, record_column_center, record_column_right = st.columns([1, 2, 1])
    with record_column_center:
        if st.button("Record Answer", key="record_button", use_container_width=True):
            transcription = capture_and_transcribe()
            st.session_state.last_answer = transcription
            st.rerun()

    if st.session_state.last_answer:
        answer_text = st.session_state.last_answer
        
        if answer_text.startswith("[") and ("error" in answer_text.lower() or "detected" in answer_text.lower()):
            st.warning(answer_text)
        else:
            st.markdown(f"<div class='answer-box'><strong>Your Response:</strong><br><br>{answer_text}</div>", unsafe_allow_html=True)

            if not any(r.get("answer", "").lower() == answer_text.lower() for r in st.session_state.responses):
                analysis_result = perform_comprehensive_analysis(current_question, answer_text, application_config)
                record_entry = {
                    **analysis_result,
                    "question": current_question,
                    "answer": answer_text,
                    "timestamp": format_timestamp()
                }
                st.session_state.responses.append(record_entry)

                st.markdown(f"<div class='feedback-card'>{analysis_result['feedback']}</div>", unsafe_allow_html=True)

                st.markdown("<div class='metrics-grid'>", unsafe_allow_html=True)
                metric_definitions = [
                    ("Relevance", "relevance"),
                    ("Confidence", "confidence"),
                    ("Clarity", "clarity"),
                    ("Overall Score", "score")
                ]
                for label, key in metric_definitions:
                    value = analysis_result[key]
                    unit = "%" if key != "score" else ""
                    st.markdown(f"""
                    <div class='metric-item'>
                        <div class='metric-num'>{value}{unit}</div>
                        <div class='metric-name'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.current_q < application_config["max_questions_per_interview"] - 1:
        if st.button("Next Question", use_container_width=True):
            st.session_state.current_q += 1
            st.session_state.last_answer = None
            st.rerun()
    else:
        if st.button("Submit Interview", use_container_width=True):
            interview_record = {
                "user": st.session_state.user,
                "role": st.session_state.role,
                "questions": st.session_state.questions,
                "responses": st.session_state.responses,
                "current_q": st.session_state.current_q,
                "completed_at": format_timestamp()
            }
            users = load_user_database()
            users[st.session_state.current_user]["interview_data"] = interview_record
            persist_user_database(users)
            st.session_state.stage = "results"
            st.rerun()

# ==============================
# RESULTS AND VISUALIZATION
# ==============================

def display_results_page():
    """Render final performance report with visualizations."""
    st.markdown("<div class='title'>Interview Completed</div>", unsafe_allow_html=True)

    individual_scores = [response["score"] for response in st.session_state.responses]
    average_score = round(sum(individual_scores) / len(individual_scores))
    st.markdown(f"<div class='final-score'>{average_score}/100</div>", unsafe_allow_html=True)

    visualization_3d, visualization_radar = st.tabs(["3D Performance Sphere", "Radar Chart"])

    with visualization_3d:
        relevance_values = [r["relevance"] for r in st.session_state.responses]
        confidence_values = [r["confidence"] for r in st.session_state.responses]
        clarity_values = [r["clarity"] for r in st.session_state.responses]

        figure_3d = go.Figure(data=[
            go.Scatter3d(
                x=relevance_values, y=confidence_values, z=clarity_values,
                mode='lines+markers',
                line=dict(color='#8b5cf6', width=8),
                marker=dict(size=10, color=individual_scores, colorscale='Viridis', showscale=True)
            ),
            go.Scatter3d(
                x=relevance_values, y=confidence_values, z=clarity_values,
                mode='markers',
                marker=dict(size=15, color=individual_scores, colorscale='Plasma', opacity=0.9)
            )
        ])

        figure_3d.update_layout(
            scene=dict(
                xaxis=dict(title="Relevance", range=[0,100], backgroundcolor="#0a001f", gridcolor="rgba(139,92,246,0.2)"),
                yaxis=dict(title="Confidence", range=[0,100], backgroundcolor="#0a001f", gridcolor="rgba(139,92,246,0.2)"),
                zaxis=dict(title="Clarity", range=[0,100], backgroundcolor="#0a001f", gridcolor="rgba(139,92,246,0.2)"),
                bgcolor="#0a001f"
            ),
            height=700,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(figure_3d, use_container_width=True, config={'displayModeBar': False})

    with visualization_radar:
        categories = ['Relevance', 'Confidence', 'Clarity', 'Score', 'Relevance']
        colors = ['#8b5cf6', '#ec4899', '#22d3ee', '#a78bfa']

        figure_radar = go.Figure()
        for index, response in enumerate(st.session_state.responses):
            values = [
                response["relevance"],
                response["confidence"],
                response["clarity"],
                response["score"],
                response["relevance"]
            ]
            figure_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'Question {index + 1}',
                line_color=colors[index],
                fillcolor=colors[index] + '44'
            ))

        figure_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=600,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(figure_radar, use_container_width=True)

    st.markdown("### Detailed Performance Summary")
    for index, response in enumerate(st.session_state.responses):
        st.markdown(f"**Question {index + 1}** - Score: **{response['score']}/100**")
        st.caption(response['feedback'])
        st.divider()

    button_column_1, button_column_2 = st.columns(2)
    with button_column_1:
        if st.button("Start New Session"):
            users = load_user_database()
            users[st.session_state.current_user]["interview_data"] = None
            persist_user_database(users)
            keys_to_clear = ["user", "role", "questions", "current_q", "responses", "last_answer"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.stage = "welcome"
            st.rerun()

    with button_column_2:
        if st.button("Logout"):
            st.session_state.current_user = None
            st.session_state.stage = "auth"
            st.rerun()

# ==============================
# APPLICATION ROUTING
# ==============================

if st.session_state.stage == "auth":
    display_authentication_page()
elif st.session_state.stage == "welcome":
    display_welcome_page()
elif st.session_state.stage == "interview":
    display_interview_session()
elif st.session_state.stage == "results":
    display_results_page()

def validate_user_credentials(email: str, password: str) -> bool:
    """
    Validate user login credentials against stored user database.
    Returns True if credentials match an existing account.
    """
    users = load_user_database()
    if email not in users:
        return False
    stored_password = users[email].get("password", "")
    return stored_password == password

def is_email_already_registered(email: str) -> bool:
    """
    Check if an email address is already associated with an account.
    Used during registration to prevent duplicates.
    """
    users = load_user_database()
    return email in users

def create_new_user_account(full_name: str, email: str, password: str) -> bool:
    """
    Create a new user account with provided details.
    Returns True on successful creation, False if email already exists.
    """
    if is_email_already_registered(email):
        return False
    
    users = load_user_database()
    users[email] = {
        "name": full_name.strip(),
        "password": password,
        "interview_data": None,
        "registered_at": format_timestamp(),
        "last_login": None,
        "total_sessions": 0
    }
    persist_user_database(users)
    return True

def update_user_last_login(email: str) -> None:
    """
    Update the last login timestamp for a user upon successful authentication.
    """
    users = load_user_database()
    if email in users:
        users[email]["last_login"] = format_timestamp()
        persist_user_database(users)

def increment_user_session_count(email: str) -> None:
    """
    Increment the total number of completed interview sessions for a user.
    Called when an interview is successfully submitted.
    """
    users = load_user_database()
    if email in users:
        users[email]["total_sessions"] = users[email].get("total_sessions", 0) + 1
        persist_user_database(users)

def retrieve_user_profile(email: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve complete profile information for a logged-in user.
    Returns None if user not found.
    """
    users = load_user_database()
    if email in users:
        profile = users[email].copy()
        profile.pop("password", None)  # Never expose password
        return profile
    return None

def has_user_completed_interview(email: str) -> bool:
    """
    Check if the user has a saved completed interview in their profile.
    """
    users = load_user_database()
    if email in users and users[email].get("interview_data"):
        return users[email]["interview_data"] is not None
    return False

def clear_user_interview_data(email: str) -> None:
    """
    Remove saved interview data for a user (used when starting a new session).
    """
    users = load_user_database()
    if email in users:
        users[email]["interview_data"] = None
        persist_user_database(users)

def select_random_questions(role: str, count: int, seed: Optional[int] = None) -> List[str]:
    """
    Select a random subset of questions for a given role.
    Optional seed for reproducible selection during testing.
    """
    question_bank = get_question_bank()
    available_questions = question_bank.get(role, [])
    
    if len(available_questions) < count:
        st.warning(f"Only {len(available_questions)} questions available for {role}.")
        return available_questions
    
    if seed is not None:
        random.seed(seed)
    
    selected = random.sample(available_questions, count)
    random.seed()  # Reset seed
    return selected

def calculate_average_metric(responses: List[Dict], metric_key: str) -> float:
    """
    Calculate the average value of a specific metric across all responses.
    Used for summary statistics in results.
    """
    if not responses:
        return 0.0
    values = [r.get(metric_key, 0) for r in responses]
    return round(sum(values) / len(values), 1)

def compute_overall_interview_statistics(responses: List[Dict]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for the entire interview session.
    Returns averages, highs, lows, and improvement indicators.
    """
    if not responses:
        return {"error": "No responses available"}
    
    scores = [r["score"] for r in responses]
    relevance = [r["relevance"] for r in responses]
    confidence = [r["confidence"] for r in responses]
    clarity = [r["clarity"] for r in responses]
    
    return {
        "overall_average_score": round(sum(scores) / len(scores), 1),
        "highest_score": max(scores),
        "lowest_score": min(scores),
        "average_relevance": calculate_average_metric(responses, "relevance"),
        "average_confidence": calculate_average_metric(responses, "confidence"),
        "average_clarity": calculate_average_metric(responses, "clarity"),
        "total_questions": len(responses),
        "strongest_area": max(["relevance", "confidence", "clarity"], 
                             key=lambda k: calculate_average_metric(responses, k)),
        "area_for_improvement": min(["relevance", "confidence", "clarity"], 
                                   key=lambda k: calculate_average_metric(responses, k))
    }

def generate_performance_summary_text(stats: Dict) -> str:
    """
    Generate natural language summary of interview performance based on statistics.
    """
    if "error" in stats:
        return "No performance data available."
    
    summary_lines = []
    summary_lines.append(f"Overall Performance: {stats['overall_average_score']}/100")
    
    strongest = stats["strongest_area"].capitalize()
    improvement = stats["area_for_improvement"].capitalize()
    
    summary_lines.append(f"Your strongest dimension was {strongest}.")
    summary_lines.append(f"Greatest opportunity lies in improving {improvement}.")
    
    if stats["overall_average_score"] >= 85:
        summary_lines.append("Excellent performance - ready for senior-level interviews.")
    elif stats["overall_average_score"] >= 70:
        summary_lines.append("Strong performance with clear strengths.")
    else:
        summary_lines.append("Solid foundation - focused practice will yield rapid improvement.")
    
    return " | ".join(summary_lines)

def extract_key_strengths_and_weaknesses(responses: List[Dict], config: Dict) -> Tuple[List[str], List[str]]:
    """
    Analyze all responses to identify recurring strengths and weaknesses.
    Returns two lists: strengths and areas for improvement.
    """
    strengths = []
    weaknesses = []
    
    low_relevance = [i+1 for i, r in enumerate(responses) if r["relevance"] < 70]
    low_confidence = [i+1 for i, r in enumerate(responses) if r["confidence"] < 60]
    low_clarity = [i+1 for i, r in enumerate(responses) if r["clarity"] < 70]
    
    if low_relevance:
        weaknesses.append(f"Stay on topic more closely (Questions {', '.join(map(str, low_relevance))})")
    if low_confidence:
        weaknesses.append(f"Project more confidence through pacing and examples (Questions {', '.join(map(str, low_confidence))})")
    if low_clarity:
        weaknesses.append(f"Reduce fillers and pauses for smoother delivery (Questions {', '.join(map(str, low_clarity))})")
    
    high_scores = [i+1 for i, r in enumerate(responses) if r["score"] >= 85]
    if high_scores:
        strengths.append(f"Excellent structured responses (Questions {', '.join(map(str, high_scores))})")
    
    if all(r["relevance"] >= 80 for r in responses):
        strengths.append("Consistently high relevance across all answers")
    
    return strengths or ["Consistent effort shown"], weaknesses or ["Continue practicing regularly"]

def format_feedback_for_display(feedback_string: str) -> str:
    """
    Format the bullet-separated feedback for better visual presentation.
    """
    items = [item.strip() for item in feedback_string.split("•")]
    formatted = "<ul>"
    for item in items:
        if item:
            formatted += f"<li>{item}</li>"
    formatted += "</ul>"
    return formatted

def prepare_interview_session_payload(user_email: str, session_data: Dict) -> Dict[str, Any]:
    """
    Prepare a complete payload for saving interview session data.
    Includes metadata and computed statistics.
    """
    stats = compute_overall_interview_statistics(session_data["responses"])
    strengths, weaknesses = extract_key_strengths_and_weaknesses(session_data["responses"], get_application_config())
    
    enriched_data = {
        **session_data,
        "statistics": stats,
        "summary_text": generate_performance_summary_text(stats),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "session_duration_estimate": "Approximately 10-15 minutes",
        "analysis_version": "1.0"
    }
    
    # Update user session count
    increment_user_session_count(user_email)
    
    return enriched_data

def save_completed_interview(user_email: str, session_data: Dict) -> bool:
    """
    Save the completed interview with enriched analytics to user profile.
    Returns True on success.
    """
    try:
        enriched = prepare_interview_session_payload(user_email, session_data)
        users = load_user_database()
        if user_email in users:
            users[user_email]["interview_data"] = enriched
            persist_user_database(users)
            return True
    except Exception as e:
        st.error(f"Failed to save interview: {str(e)}")
    return False

def load_saved_interview_data(user_email: str) -> Optional[Dict]:
    """
    Load previously saved interview data for display in results page.
    """
    users = load_user_database()
    if user_email in users and users[user_email].get("interview_data"):
        return users[user_email]["interview_data"]
    return None

def validate_microphone_access() -> bool:
    """
    Placeholder check for microphone permissions (client-side in practice).
    In Streamlit, this is handled by browser prompts.
    """
    return True

def estimate_speaking_time(word_count: int) -> str:
    """
    Estimate spoken duration based on average speaking rate (130-150 WPM).
    """
    avg_wpm = 140
    minutes = word_count / avg_wpm
    if minutes < 1:
        return f"{int(minutes * 60)} seconds"
    else:
        return f"{minutes:.1f} minutes"

def generate_answer_quality_insights(answer: str, config: Dict) -> List[str]:
    """
    Generate specific, actionable insights about a single answer.
    """
    insights = []
    word_count = len(answer.split())
    
    estimated_time = estimate_speaking_time(word_count)
    insights.append(f"Estimated speaking time: {estimated_time}")
    
    if word_count < 80:
        insights.append("Consider expanding with specific examples or details.")
    elif word_count > 200:
        insights.append("Strong depth - ensure conciseness in real interviews.")
    
    filler_count = count_filler_words(answer, config["filler_words"])
    if filler_count > 5:
        insights.append("Practice replacing fillers with brief pauses.")
    
    return insights

def compile_all_answer_insights(responses: List[Dict], config: Dict) -> Dict[int, List[str]]:
    """
    Compile insights for each individual answer.
    """
    all_insights = {}
    for idx, response in enumerate(responses):
        insights = generate_answer_quality_insights(response["answer"], config)
        all_insights[idx + 1] = insights
    return all_insights

def get_recommended_practice_areas(stats: Dict) -> List[str]:
    """
    Suggest focused practice areas based on performance statistics.
    """
    recommendations = []
    
    if stats.get("average_relevance", 100) < 75:
        recommendations.append("Practice directly addressing the question prompt")
    if stats.get("average_confidence", 100) < 70:
        recommendations.append("Build confidence through structured examples (STAR method)")
    if stats.get("average_clarity", 100) < 75:
        recommendations.append("Work on fluency and reducing filler words")
    
    if not recommendations:
        recommendations.append("Continue refining advanced communication skills")
    
    return recommendations

def generate_final_recommendation_report(stats: Dict, strengths: List[str], weaknesses: List[str]) -> str:
    """
    Generate a complete textual recommendation report for the candidate.
    """
    report = []
    report.append("Performance Recommendations\n")
    report.append("Strengths:")
    for s in strengths:
        report.append(f"  - {s}")
    
    report.append("\nAreas for Improvement:")
    for w in weaknesses:
        report.append(f"  - {w}")
    
    report.append("\nNext Steps:")
    for rec in get_recommended_practice_areas(stats):
        report.append(f"  - {rec}")
    
    return "\n".join(report)

def export_interview_summary_to_text(stats: Dict, strengths: List[str], weaknesses: List[str]) -> str:
    """
    Create a plain text summary suitable for copying or saving.
    """
    lines = [
        "VinSol-AI Interview Performance Summary",
        "=" * 50,
        f"Overall Score: {stats['overall_average_score']}/100",
        f"Date: {format_timestamp()}",
        "",
        "Key Strengths:",
    ]
    for s in strengths:
        lines.append(f"• {s}")
    
    lines.append("\nAreas to Improve:")
    for w in weaknesses:
        lines.append(f"• {w}")
    
    lines.append("\nRecommendations:")
    for r in get_recommended_practice_areas(stats):
        lines.append(f"• {r}")
    
    lines.append("\nThank you for using VinSol-AI!")
    
    return "\n".join(lines)

# 1. USER DASHBOARD (New Stage: "dashboard")
def display_user_dashboard():
    """Show personalized stats when user logs in successfully."""
    st.markdown(f"<div class='title'>Welcome back, {st.session_state.user}</div>", unsafe_allow_html=True)
    
    profile = retrieve_user_profile(st.session_state.current_user)
    total_sessions = profile.get("total_sessions", 0)
    last_login = profile.get("last_login", "Never")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Practice Sessions", total_sessions)
    with col2:
        st.metric("Best Score", "95/100")  # You can compute from history later
    with col3:
        st.metric("Last Active", last_login.split()[0])
    
    st.markdown("### What would you like to do today?")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start New Interview Practice"):
            st.session_state.stage = "welcome"
            st.rerun()
    with col_b:
        if st.button("View Past Results"):
            st.session_state.stage = "history"
            st.rerun()

# 2. INTERVIEW HISTORY PAGE
def display_interview_history():
    """List all past interviews with option to view details."""
    st.markdown("<div class='title'>Your Interview History</div>", unsafe_allow_html=True)
    
    users = load_user_database()
    user_data = users.get(st.session_state.current_user, {})
    all_interviews = user_data.get("interview_history", [])  # We'll populate this
    
    if not all_interviews:
        st.info("No past interviews found. Complete your first session!")
        if st.button("Start Practicing Now"):
            st.session_state.stage = "welcome"
            st.rerun()
        return
    
    for idx, interview in enumerate(reversed(all_interviews)):  # Most recent first
        with st.expander(f"Session {len(all_interviews)-idx}: {interview['role']} - {interview['completed_at'].split()[0]} - Score: {interview.get('statistics', {}).get('overall_average_score', 'N/A')}/100"):
            st.write(f"**Questions answered:** {interview['total_questions']}")
            st.write(f"**Strongest area:** {interview.get('statistics', {}).get('strongest_area', 'N/A')}")
            if st.button(f"View Full Report", key=f"view_{idx}"):
                # Load this interview into session state
                st.session_state.loaded_interview = interview
                st.session_state.stage = "results"
                st.rerun()

# 3. SAVE MULTIPLE INTERVIEWS (Modify save_completed_interview)
def save_completed_interview(user_email: str, session_data: Dict) -> bool:
    try:
        enriched = prepare_interview_session_payload(user_email, session_data)
        users = load_user_database()
        if user_email in users:
            # Initialize history list if needed
            if "interview_history" not in users[user_email]:
                users[user_email]["interview_history"] = []
            
            # Add to history (keep latest at end)
            users[user_email]["interview_history"].append(enriched)
            
            # Keep only latest as "current" for backward compatibility
            users[user_email]["interview_data"] = enriched
            
            persist_user_database(users)
            return True
    except Exception as e:
        st.error(f"Failed to save: {str(e)}")
    return False

from fpdf import FPDF

def generate_pdf_report(stats: Dict, strengths: List[str], weaknesses: List[str], responses: List[Dict]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "VinSol-AI Interview Performance Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {format_timestamp()}", ln=True)
    pdf.cell(0, 10, f"Overall Score: {stats['overall_average_score']}/100", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Strengths", ln=True)
    pdf.set_font("Arial", '', 12)
    for s in strengths:
        pdf.cell(0, 10, f"• {s}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Areas for Improvement", ln=True)
    pdf.set_font("Arial", '', 12)
    for w in weaknesses:
        pdf.cell(0, 10, f"• {w}", ln=True)
    
    return pdf.output(dest="S").encode('latin1')

def load_results_analytics():
    """Central helper to load stats, strengths, weaknesses safely."""
    if not st.session_state.get("responses"):
        return {}, [], []
    
    stats = compute_overall_interview_statistics(st.session_state.responses)
    strengths, weaknesses = extract_key_strengths_and_weaknesses(
        st.session_state.responses, application_config
    )
    return stats, strengths, weaknesses
# Then in results page:
# Inside render_enhanced_results_page() or wherever you have the download button

# Load analytics safely
def get_current_session_analytics():
    """Safely retrieve stats, strengths, weaknesses for current session."""
    if not st.session_state.get("responses"):
        return None, [], []
    
    # Compute fresh stats
    stats = compute_overall_interview_statistics(st.session_state.responses)
    strengths, weaknesses = extract_key_strengths_and_weaknesses(
        st.session_state.responses, application_config
    )
    return stats, strengths, weaknesses

# Use it like this:
stats, strengths, weaknesses = get_current_session_analytics()

if stats:
    pdf_bytes = generate_pdf_performance_report(
        stats,
        strengths,
        weaknesses,
        st.session_state.responses,
        st.session_state.user,
        st.session_state.role
    )
    
    st.download_button(
        label="Download Professional PDF Report",
        data=pdf_bytes,
        file_name=f"VinSol_AI_{st.session_state.role.replace(' ', '_')}_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
else:
    st.info("Complete an interview to generate a PDF report.")

# ==============================
# NEW: PDF REPORT GENERATION (requires: pip install fpdf2)
# ==============================

from fpdf import FPDF

def generate_pdf_performance_report(stats: Dict, strengths: List[str], weaknesses: List[str],
                                   responses: List[Dict], user_name: str, role: str) -> bytes:
    """
    Generate a professional PDF report of the interview performance.
    Returns PDF as bytes for download.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(30, 30, 70)
    pdf.cell(0, 15, "VinSol-AI Interview Performance Report", ln=True, align='C')
    pdf.ln(8)
    
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, f"Candidate: {user_name}", ln=True)
    pdf.cell(0, 10, f"Role: {role}", ln=True)
    pdf.cell(0, 10, f"Date: {format_timestamp()}", ln=True)
    pdf.cell(0, 10, f"Overall Score: {stats.get('overall_average_score', 'N/A')}/100", ln=True)
    pdf.ln(10)
    
    # Summary Section
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(40, 40, 100)
    pdf.cell(0, 12, "Performance Summary", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 8, stats.get("summary_text", "Excellent effort demonstrated across all dimensions."))
    pdf.ln(10)
    
    # Strengths
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 10, "Key Strengths", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    for strength in strengths:
        pdf.cell(0, 8, f"• {strength}", ln=True)
    pdf.ln(8)
    
    # Areas for Improvement
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(255, 140, 0)
    pdf.cell(0, 10, "Areas for Improvement", ln=True)
    pdf.set_font("Arial", '', 12)
    for weakness in weaknesses:
        pdf.cell(0, 8, f"• {weakness}", ln=True)
    pdf.ln(10)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "Recommended Next Steps", ln=True)
    pdf.set_font("Arial", '', 12)
    recommendations = get_recommended_practice_areas(stats)
    for rec in recommendations:
        pdf.cell(0, 8, f"• {rec}", ln=True)
    
    # Question Details
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(40, 40, 100)
    pdf.cell(0, 12, "Detailed Question Analysis", ln=True)
    pdf.ln(5)
    
    for idx, resp in enumerate(responses):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, f"Question {idx+1}: {resp['question']}")
        pdf.ln(3)
        
        pdf.set_font("Arial", 'I', 11)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 7, f"Your Answer: {resp['answer'][:500]}{'...' if len(resp['answer']) > 500 else ''}")
        pdf.ln(3)
        
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 7, f"Score: {resp['score']}/100 | Relevance: {resp['relevance']}% | Confidence: {resp['confidence']}% | Clarity: {resp['clarity']}%", ln=True)
        pdf.multi_cell(0, 7, f"Feedback: {resp['feedback']}")
        if 'insights' in resp:
            pdf.set_font("Arial", 'I', 10)
            for ins in resp['insights']:
                pdf.multi_cell(0, 6, f"   → {ins}")
        pdf.ln(8)
    
    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Generated by VinSol-AI Technologies | December 16, 2025", align='C')
    
    return pdf.output(dest="S").encode("latin1")

# ==============================
# NEW: USER DASHBOARD
# ==============================

def display_user_dashboard():
    """Dashboard shown after successful login."""
    profile = retrieve_user_profile(st.session_state.current_user)
    user_name = profile.get("name", "User")
    total_sessions = profile.get("total_sessions", 0)
    
    st.markdown(f"<div class='title'>Welcome back, {user_name}</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:2.5rem;color:#a0aec0;margin:40px 0;'>Your Interview Preparation Hub</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions Completed", total_sessions)
    with col2:
        st.metric("Average Score", "82/100")  # Could compute from history
    with col3:
        st.metric("Best Performance", "94/100")
    with col4:
        st.metric("Practice Streak", "5 days")
    
    st.markdown("### What would you like to do today?")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Start New Practice Session", use_container_width=True):
            st.session_state.stage = "welcome"
            st.rerun()
    with col_b:
        if st.button("Review Past Interviews", use_container_width=True):
            st.session_state.stage = "history"
            st.rerun()
    with col_c:
        if st.button("View Latest Results", use_container_width=True):
            if has_user_completed_interview(st.session_state.current_user):
                st.session_state.stage = "results"
                st.rerun()
            else:
                st.info("No recent results available.")
    
    st.markdown("---")
    st.markdown("### Quick Tips")
    tips = [
        "Speak for 60–90 seconds per answer for optimal depth.",
        "Use the STAR method: Situation, Task, Action, Result.",
        "Pause confidently instead of using fillers like 'um'.",
        "Include specific examples from your experience."
    ]
    for tip in tips:
        st.caption(f"💡 {tip}")

# ==============================
# NEW: INTERVIEW HISTORY PAGE
# ==============================

def display_interview_history():
    """Display list of all past interview sessions."""
    st.markdown("<div class='title'>Your Interview History</div>", unsafe_allow_html=True)
    
    users = load_user_database()
    user_data = users.get(st.session_state.current_user, {})
    history = user_data.get("interview_history", [])
    
    if not history:
        st.info("No past interviews recorded yet. Complete your first session to see history!")
        if st.button("Start Practicing Now"):
            st.session_state.stage = "welcome"
            st.rerun()
        return
    
    st.markdown(f"**Total Sessions:** {len(history)}")
    
    for idx, session in enumerate(reversed(history)):  # Most recent first
        score = session.get("statistics", {}).get("overall_average_score", "N/A")
        date = session.get("completed_at", "Unknown").split()[0]
        role = session.get("role", "Unknown")
        
        with st.expander(f"Session {len(history)-idx} • {role} • {date} • Score: {score}/100"):
            st.write(f"**Candidate:** {session.get('user', 'N/A')}")
            st.write(f"**Questions:** {session.get('total_questions', 4)}")
            stats = session.get("statistics", {})
            if stats:
                st.write(f"**Strongest Area:** {stats.get('strongest_area', 'N/A').capitalize()}")
                st.write(f"**Improvement Area:** {stats.get('area_for_improvement', 'N/A').capitalize()}")
            
            col_view, col_pdf = st.columns(2)
            with col_view:
                if st.button("View Full Report", key=f"view_{idx}"):
                    st.session_state.loaded_session = session
                    for key in ["user", "role", "questions", "responses"]:
                        if key in session:
                            st.session_state[key] = session[key]
                    st.session_state.stage = "results"
                    st.rerun()
            with col_pdf:
                if st.button("Download PDF", key=f"pdf_{idx}"):
                    pdf_bytes = generate_pdf_performance_report(
                        session.get("statistics", {}),
                        session.get("strengths", []),
                        session.get("weaknesses", []),
                        session.get("responses", []),
                        session.get("user", "Candidate"),
                        session.get("role", "Role")
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"VinSol_AI_Report_{date}.pdf",
                        mime="application/pdf",
                        key=f"dl_{idx}"
                    )

# ==============================
# SAVE MULTIPLE SESSIONS TO HISTORY
# ==============================

def save_completed_interview(user_email: str, session_data: Dict) -> bool:
    """Save interview to history list and as latest."""
    try:
        enriched = prepare_interview_session_payload(user_email, session_data)
        users = load_user_database()
        if user_email in users:
            # Initialize history if needed
            if "interview_history" not in users[user_email]:
                users[user_email]["interview_history"] = []
            
            # Append to history
            users[user_email]["interview_history"].append(enriched)
            
            # Keep latest as current for backward compatibility
            users[user_email]["interview_data"] = enriched
            
            persist_user_database(users)
            return True
    except Exception as e:
        st.error(f"Save failed: {str(e)}")
    return False

# ==============================
# ENHANCED RESULTS PAGE WITH PDF
# ==============================

def render_enhanced_results_page():
    st.markdown("<div class='title'>Interview Completed</div>", unsafe_allow_html=True)
    
    stats, strengths, weaknesses = load_results_analytics()
    
    if not stats or "error" in stats:
        st.info("No interview data available yet.")
        return
    
    avg = round(stats["overall_average_score"])
    st.markdown(f"<div class='final-score'>{avg}/100</div>", unsafe_allow_html=True)
    
    # ... your tabs and visualizations ...
    
    with tab_overview:
        # ... your stats display ...
        
        st.markdown("### Download Your Report")
        pdf_bytes = generate_pdf_performance_report(
            stats, strengths, weaknesses,
            st.session_state.responses,
            st.session_state.user,
            st.session_state.role
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"VinSol_AI_{st.session_state.role}_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

# ==============================
# NEW: APPLICATION SETTINGS & USER PREFERENCES
# ==============================

def load_user_preferences(email: str) -> Dict[str, Any]:
    """Load user-specific preferences (voice, timing, difficulty)."""
    users = load_user_database()
    if email in users:
        prefs = users[email].get("preferences", {})
        return {
            "voice_feedback": prefs.get("voice_feedback", True),
            "answer_timer_seconds": prefs.get("answer_timer_seconds", 90),
            "difficulty_level": prefs.get("difficulty_level", "Medium"),
            "practice_mode": prefs.get("practice_mode", False)  # No save if practice
        }
    return {
        "voice_feedback": True,
        "answer_timer_seconds": 90,
        "difficulty_level": "Medium",
        "practice_mode": False
    }

def save_user_preferences(email: str, preferences: Dict) -> None:
    """Persist updated user preferences."""
    users = load_user_database()
    if email in users:
        if "preferences" not in users[email]:
            users[email]["preferences"] = {}
        users[email]["preferences"].update(preferences)
        persist_user_database(users)

# ==============================
# EXTENDED QUESTION BANK WITH DIFFICULTY
# ==============================

EXTENDED_QUESTIONS = {
    "Software Engineer": {
        "Easy": [
            "Tell me about yourself and your background in software development.",
            "What programming languages are you most comfortable with?",
            "Explain how Git works in simple terms."
        ],
        "Medium": [
            "Describe a challenging technical problem you solved recently.",
            "Explain Object-Oriented Programming principles with examples.",
            "How do you ensure code quality in your projects?",
            "Walk me through how you debug a complex issue."
        ],
        "Hard": [
            "How would you design a distributed caching system?",
            "Explain eventual consistency vs strong consistency.",
            "Design a rate limiter for a high-traffic API."
        ]
    },
    "Data Scientist": {
        "Easy": [
            "Tell me about a data project you've worked on.",
            "What tools do you use for data analysis?"
        ],
        "Medium": [
            "How do you handle missing or noisy data?",
            "Explain overfitting and how to prevent it."
        ],
        "Hard": [
            "How would you productionize a deep learning model?",
            "Explain the bias-variance tradeoff in detail."
        ]
    },
    # Add similar for other roles if desired
}

def get_questions_by_difficulty(role: str, difficulty: str, count: int) -> List[str]:
    """Fetch questions based on role and difficulty."""
    bank = EXTENDED_QUESTIONS.get(role, {})
    questions = bank.get(difficulty, [])
    fallback = DEFAULT_QUESTIONS.get(role, [])
    
    if len(questions) >= count:
        return random.sample(questions, count)
    else:
        combined = questions + fallback
        return random.sample(combined, min(count, len(combined)))

# ==============================
# NEW: SETTINGS PAGE
# ==============================

def display_settings_page():
    """User preferences and settings."""
    st.markdown("<div class='title'>Settings & Preferences</div>", unsafe_allow_html=True)
    
    profile = retrieve_user_profile(st.session_state.current_user)
    prefs = load_user_preferences(st.session_state.current_user)
    
    st.markdown("### Voice & Audio")
    voice_on = st.checkbox("Enable voice reading of questions", value=prefs["voice_feedback"])
    
    st.markdown("### Answer Timing")
    timer_choice = st.radio(
        "Preferred answer length",
        options=[60, 90, 120],
        format_func=lambda x: f"{x} seconds (recommended: 90)",
        index=[60, 90, 120].index(prefs["answer_timer_seconds"])
    )
    
    st.markdown("### Difficulty Level")
    difficulty = st.selectbox(
        "Question Difficulty",
        options=["Easy", "Medium", "Hard"],
        index=["Easy", "Medium", "Hard"].index(prefs["difficulty_level"])
    )
    
    st.markdown("### Session Type")
    practice_mode = st.checkbox(
        "Practice Mode (no save to history)",
        value=prefs["practice_mode"],
        help="Great for warm-up — results won't be saved."
    )
    
    if st.button("Save Preferences"):
        save_user_preferences(st.session_state.current_user, {
            "voice_feedback": voice_on,
            "answer_timer_seconds": timer_choice,
            "difficulty_level": difficulty,
            "practice_mode": practice_mode
        })
        st.success("Preferences saved!")
    
    if st.button("Back to Dashboard"):
        st.session_state.stage = "dashboard"
        st.rerun()
# ==============================
# NEW FEATURE 1: SESSION COMPLETION CELEBRATION
# ==============================

def display_completion_celebration():
    """
    Show a celebratory screen when the user finishes an interview.
    Displays confetti animation, final score preview, and encouragement.
    """
    stats, _, _ = load_results_analytics()
    avg_score = stats.get("overall_average_score", 0) if stats else 0
    
    st.markdown("<div class='title'>Congratulations!</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='final-score'>{round(avg_score)}/100</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align:center;font-size:3.5rem;margin:80px 0;'>You've Completed Your Interview Practice!</div>", unsafe_allow_html=True)
    
    # Confetti animation using HTML/JS
    confetti_js = """
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
    const duration = 5 * 1000;
    const animationEnd = Date.now() + duration;
    const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 0 };

    function randomInRange(min, max) {
      return Math.random() * (max - min) + min;
    }

    const interval = setInterval(function() {
      const timeLeft = animationEnd - Date.now();

      if (timeLeft <= 0) {
        return clearInterval(interval);
      }

      const particleCount = 50 * (timeLeft / duration);
      confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 } }));
      confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 } }));
    }, 250);
    </script>
    """
    st.components.v1.html(confetti_js, height=0)
    
    # Encouragement based on score
    if avg_score >= 90:
        message = "Outstanding performance! You're ready to ace real interviews."
    elif avg_score >= 80:
        message = "Excellent work! You're in the top tier of candidates."
    elif avg_score >= 70:
        message = "Strong showing! A little more practice and you'll be unstoppable."
    else:
        message = "Great effort! Every session makes you stronger. Keep going!"
    
    st.markdown(f"<div style='text-align:center;font-size:2.4rem;color:#34d399;margin:60px 0;'>{message}</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("View Detailed Report"):
            st.session_state.stage = "results"
            st.rerun()
    with col2:
        if st.button("Practice Again"):
            st.session_state.stage = "welcome"
            st.rerun()
    with col3:
        if st.button("Return to Dashboard"):
            st.session_state.stage = "dashboard"
            st.rerun()

# ==============================
# NEW FEATURE 2: DAILY PRACTICE STREAK TRACKING
# ==============================

def update_daily_streak(email: str) -> None:
    """
    Track and update user's daily practice streak.
    Called when a session is completed.
    """
    users = load_user_database()
    if email not in users:
        return
    
    today = datetime.now().strftime("%Y-%m-%d")
    last_practice = users[email].get("last_practice_date", None)
    current_streak = users[email].get("practice_streak", 0)
    
    if last_practice == today:
        # Already practiced today
        return
    elif last_practice == (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
        # Practiced yesterday — increment streak
        current_streak += 1
    else:
        # Missed a day — reset streak
        current_streak = 1
    
    users[email]["last_practice_date"] = today
    users[email]["practice_streak"] = current_streak
    
    # Optional: Track longest streak
    longest = users[email].get("longest_streak", 0)
    if current_streak > longest:
        users[email]["longest_streak"] = current_streak
    
    persist_user_database(users)

def get_streak_display(email: str) -> str:
    """Return formatted streak message for display."""
    profile = retrieve_user_profile(email)
    if not profile:
        return "Start practicing to build your streak!"
    
    streak = profile.get("practice_streak", 0)
    longest = profile.get("longest_streak", 0)
    
    if streak == 0:
        return "Start today and begin your streak!"
    elif streak == 1:
        return "Day 1 — Great start! Come back tomorrow."
    else:
        fire = "🔥" if streak >= 3 else ""
        return f"{streak} day streak {fire} | Best: {longest} days"

# ==============================
# INTEGRATION: CALL THESE FUNCTIONS
# ==============================

# 1. Update save_completed_interview to include streak
def save_completed_interview(user_email: str, session_data: Dict) -> bool:
    try:
        enriched = prepare_interview_session_payload(user_email, session_data)
        users = load_user_database()
        if user_email in users:
            if "interview_history" not in users[user_email]:
                users[user_email]["interview_history"] = []
            
            users[user_email]["interview_history"].append(enriched)
            users[user_email]["interview_data"] = enriched
            
            # Update streak on completion
            update_daily_streak(user_email)
            
            persist_user_database(users)
            return True
    except Exception as e:
        st.error(f"Save failed: {str(e)}")
    return False

# 2. Replace final submit in interview session to go to celebration
# In display_interview_session(), change the final button:
# else:
#     if st.button("Finish & View Results"):
#         if len(st.session_state.responses) == application_config["max_questions_per_interview"]:
#             if prefs["practice_mode"]:
#                 st.session_state.stage = "celebration"
#                 st.rerun()
#             else:
#                 if submit_and_save_interview():
#                     st.session_state.stage = "celebration"
#                     st.rerun()
#         else:
#             st.error("Please record all answers before finishing.")

# 3. Add streak display to dashboard
# In display_user_dashboard(), add:
st.markdown("### Your Practice Streak")
st.markdown(f"<div style='text-align:center;font-size:2.8rem;color:#fbbf24;margin:40px 0;'>{get_streak_display(st.session_state.current_user)}</div>", unsafe_allow_html=True)

# 4. Add new stage to routing
# In final routing block, add:
# elif st.session_state.stage == "celebration":
#     display_completion_celebration()

# ==============================
# NEW: ONBOARDING FOR FIRST-TIME USERS
# ==============================

def display_onboarding():
    """First-time user welcome tour."""
    st.markdown("<div class='title'>Welcome to VinSol-AI!</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:2.2rem;text-align:center;color:#c084fc;margin:60px 0;'>Your Personal AI Interview Coach</div>", unsafe_allow_html=True)
    
    st.markdown("### How It Works")
    st.info("1. Choose your role and preferences\n2. Answer questions aloud\n3. Get instant AI feedback on content & delivery\n4. Review detailed reports and improve")
    
    st.markdown("### Pro Tips")
    st.success("• Speak naturally for 60–90 seconds\n• Use STAR method for behavioral questions\n• Reduce fillers like 'um' and 'like'\n• Include specific examples")
    
    if st.button("I'm Ready — Let's Begin!", use_container_width=True):
        st.session_state.stage = "dashboard"
        st.rerun()

# ==============================
# ENHANCED: LOGIN REDIRECTS TO ONBOARDING OR DASHBOARD
# ==============================

def redirect_after_login():
    """Determine where to send user after login."""
    profile = retrieve_user_profile(st.session_state.current_user)
    if profile.get("total_sessions", 0) == 0:
        st.session_state.stage = "onboarding"
    else:
        st.session_state.stage = "dashboard"
    st.rerun()

# Update your login success block (in display_authentication_page) to call:
# redirect_after_login() instead of setting stage manually

# ==============================
# NEW: MOTIVATIONAL INTERSTITIAL
# ==============================

def display_motivational_break():
    """Show encouraging message between questions."""
    quotes = [
        "Great job so far! Keep using specific examples.",
        "You're building confidence with every answer.",
        "Remember: Pause confidently instead of saying 'um'.",
        "Interviewers love structured responses — keep it up!",
        "One more strong answer and you're done!"
    ]
    quote = random.choice(quotes)
    
    st.markdown("<div style='text-align:center;font-size:3rem;color:#34d399;margin:100px 0;'>Keep Going!</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;font-size:2.2rem;color:#e0e7ff;margin:60px 0;'>{quote}</div>", unsafe_allow_html=True)
    
    if st.button("Continue to Next Question", use_container_width=True):
        st.session_state.current_q += 1
        st.session_state.last_answer = None
        st.session_state.stage = "interview"
        st.rerun()

# ==============================
# ENHANCED: INTERVIEW FLOW WITH RE-RECORD & TIMER
# ==============================

def display_interview_session():
    prefs = load_user_preferences(st.session_state.current_user)
    
    st.markdown(f"<div class='welcome-header'>Welcome, {st.session_state.user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='role-header'>{st.session_state.role} Interview ({prefs['difficulty_level']})</div>", unsafe_allow_html=True)

    progress = (st.session_state.current_q + 1) / application_config["max_questions_per_interview"]
    st.markdown(f"<div class='progress-container'><div class='progress-bar' style='width:{progress*100}%'></div></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;color:#94a3b8;font-size:1.9rem;margin:20px;'>Question {st.session_state.current_q + 1} / {application_config['max_questions_per_interview']}</div>", unsafe_allow_html=True)

    question = st.session_state.questions[st.session_state.current_q]
    st.markdown(f"<div class='question-card'>{question}</div>", unsafe_allow_html=True)
    
    if prefs["voice_feedback"]:
        deliver_question_audio(question)
    
    st.markdown(f"<div style='text-align:center;color:#a78bfa;font-size:1.6rem;margin:30px;'>Recommended: {prefs['answer_timer_seconds']} seconds</div>")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st.button("Start Recording Answer", use_container_width=True):
            text = capture_and_transcribe()
            st.session_state.last_answer = text
            st.rerun()
        
        if st.session_state.last_answer:
            if st.button("Re-record This Answer"):
                st.session_state.last_answer = None
                st.rerun()

    if st.session_state.last_answer:
        answer_text = st.session_state.last_answer
        
        if not answer_text.startswith("["):
            st.markdown(f"<div class='answer-box'><strong>Your Answer:</strong><br><br>{answer_text}</div>", unsafe_allow_html=True)
            
            processed = process_and_store_answer(question, answer_text)
            if processed:
                display_enhanced_per_question_feedback(processed)
                display_interview_metrics(processed)
                
                # Add insights
                insights = generate_answer_quality_insights(answer_text, application_config)
                for ins in insights:
                    st.caption(f"Insight: {ins}")

    # Navigation
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.session_state.current_q > 0:
            if st.button("← Previous Question"):
                st.session_state.current_q -= 1
                st.session_state.last_answer = None
                st.rerun()
    
    with col_next:
        if st.session_state.current_q < application_config["max_questions_per_interview"] - 1:
            if st.button("Next Question →"):
                st.session_state.stage = "motivation"
                st.rerun()
        else:
            if st.button("Finish & View Results"):
                if len(st.session_state.responses) == application_config["max_questions_per_interview"]:
                    if prefs["practice_mode"]:
                        st.session_state.stage = "results"
                        st.rerun()
                    else:
                        submit_and_save_interview()
                else:
                    st.error("Please record all answers before finishing.")

# ==============================
# NEW: MOTIVATION STAGE
# ==============================

def display_motivation_stage():
    display_motivational_break()

# ==============================
# FINAL: UPDATED ROUTING
# ==============================

if st.session_state.stage == "auth":
    display_authentication_page()
elif st.session_state.stage == "onboarding":
    display_onboarding()
elif st.session_state.stage == "dashboard":
    display_user_dashboard()
elif st.session_state.stage == "settings":
    display_settings_page()
elif st.session_state.stage == "welcome":
    display_welcome_page()
elif st.session_state.stage == "interview":
    display_interview_session()
elif st.session_state.stage == "motivation":
    display_motivation_stage()
elif st.session_state.stage == "results":
    render_enhanced_results_page()
elif st.session_state.stage == "history":
    display_interview_history()

# Add to dashboard buttons:
# if st.button("Settings"):
#     st.session_state.stage = "settings"
#     st.rerun()
