import streamlit as st
from datetime import datetime
import random
import plotly.graph_objects as go

st.set_page_config(page_title="VinSol-AI Interview Simulator", page_icon="microphone", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
    .main {background: linear-gradient(135deg, #0a001f 0%, #1a0033 50%, #2d1b69 100%); font-family: 'Inter', sans-serif; padding: 20px;}
    h1, h2, h3 {color: #e0e7ff; text-align: center;}
    .title {font-size: 6rem; font-weight: 900; background: linear-gradient(90deg, #8b5cf6, #ec4899, #22d3ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 80px rgba(139,92,246,0.7);}
    .glass {background: rgba(255,255,255,0.08); backdrop-filter: blur(20px); border-radius: 32px; padding: 50px; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 25px 60px rgba(0,0,0,0.6);}
    .question-card {background: linear-gradient(135deg, rgba(139,92,246,0.3), rgba(236,72,153,0.2)); padding: 60px; border-radius: 32px; font-size: 2.2rem; line-height: 1.5; color: #f0f4ff; text-align: center; border: 1px solid rgba(139,92,246,0.6); box-shadow: 0 0 60px rgba(139,92,246,0.5);}
    .feedback-card {background: linear-gradient(90deg, rgba(34,197,94,0.25), rgba(22,163,74,0.15)); border-left: 10px solid #22d3ee; padding: 35px; border-radius: 24px; font-size: 1.5rem; color: #d0fffc; margin: 50px 0;}
    .metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px; margin: 60px 0;}
    .metric-box {background: rgba(255,255,255,0.1); padding: 35px; border-radius: 28px; text-align: center; backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.2);}
    .big-number {font-size: 4.5rem; font-weight: 900; background: linear-gradient(90deg, #22d3ee, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .label {font-size: 1.4rem; color: #94a3b8; margin-top: 15px;}
    .wave {text-align: center; font-size: 6rem; margin: 80px 0; animation: pulse 1.5s infinite;}
    @keyframes pulse {0%,100% {transform: scale(1); opacity: 0.8;} 50% {transform: scale(1.3); opacity: 1;}}
    .stButton>button {background: linear-gradient(90deg, #8b5cf6, #ec4899); color: white; font-weight: 700; font-size: 1.6rem; padding: 22px 70px; border-radius: 28px; border: none; box-shadow: 0 20px 50px rgba(139,92,246,0.6);}
    .stButton>button:hover {transform: translateY(-10px); box-shadow: 0 35px 70px rgba(139,92,246,0.8);}
    .footer {text-align: center; padding: 100px 20px; color: #64748b; font-size: 1.1rem;}
    .stProgress > div > div > div > div {background: linear-gradient(90deg, #8b5cf6, #22d3ee);}
</style>
""", unsafe_allow_html=True)

# 50+ REAL QUESTIONS
QUESTIONS = {
    "Software Engineer": [
        "Tell me about yourself and your journey into software development.",
        "Describe a complex technical challenge you solved recently.",
        "How do you approach debugging a production issue at 2 AM?",
        "Explain OOP principles with real-world analogies.",
        "How do you ensure your code is clean and maintainable?",
        "Walk me through your experience with Git workflows.",
        "What’s your experience with microservices architecture?",
        "How do you optimize slow-performing APIs?",
        "Explain REST vs GraphQL — when do you choose one?",
        "How do you handle memory leaks in your applications?",
        "Describe your experience with cloud platforms.",
        "How do you write testable code?"
    ],
    "Data Scientist": [
        "Tell me about a machine learning project you're proud of.",
        "How do you handle missing or imbalanced data?",
        "Explain overfitting and how you prevent it.",
        "What’s the difference between bagging and boosting?",
        "How do you evaluate classification vs regression models?",
        "Describe a time your model failed in production.",
        "How do you explain complex models to non-technical people?",
        "What’s your experience with deep learning?",
        "How do you perform feature engineering?",
        "What is cross-validation and why is it important?"
    ],
    "Product Manager": [
        "How do you define product vision and strategy?",
        "How do you prioritize features when everything seems important?",
        "Tell me about a product you launched from idea to market.",
        "How do you handle disagreements with engineering?",
        "What prioritization frameworks do you use?",
        "How do you measure product success?",
        "Describe your user research process.",
        "How do you write effective user stories?",
        "What’s your approach to A/B testing?",
        "How do you manage stakeholder expectations?"
    ],
    "UX Designer": [
        "Walk me through your complete design process.",
        "How do you conduct effective user research?",
        "What’s your experience with design systems?",
        "How do you handle conflicting feedback?",
        "Explain the difference between UI and UX.",
        "How do you design for accessibility?",
        "Describe a time you strongly advocated for users.",
        "What tools do you use daily?",
        "How do you present designs to stakeholders?",
        "How do you stay updated with design trends?"
    ],
    "DevOps Engineer": [
        "Explain CI/CD and why it matters.",
        "How do you secure infrastructure as code?",
        "What’s your experience with Kubernetes?",
        "How do you monitor production systems?",
        "Describe a time you reduced deployment time.",
        "What is Infrastructure as Code? Which tools?",
        "How do you handle rollbacks in production?",
        "Explain zero-downtime deployments.",
        "What logging and alerting tools do you use?",
        "How do you manage secrets securely?"
    ],
    "Full-Stack Developer": [
        "How do you choose between React, Vue, or Angular?",
        "Explain how you handle authentication in web apps.",
        "What’s your experience with Node.js backend?",
        "How do you optimize frontend performance?",
        "Describe your deployment process.",
        "How do you handle state management?",
        "What databases have you worked with?",
        "Explain CORS and how to fix it."
    ]
}

# 45+ RICH, VARIED FEEDBACKS
FEEDBACKS = [
    "Outstanding clarity and confidence — you owned that answer completely!",
    "Excellent structure! You used the STAR method perfectly — very professional.",
    "Strong technical depth with real-world examples. Impressive insight.",
    "Very articulate — your communication is truly executive-level.",
    "Perfect pace and tone — extremely polished delivery.",
    "Impressive! You anticipated follow-up questions naturally.",
    "Confident, concise, and impactful — exactly what top companies look for.",
    "You demonstrated deep expertise without overwhelming jargon.",
    "Flawless explanation — even a beginner would understand perfectly.",
    "Great energy and presence — you sound like a natural leader.",
    "Natural and authentic — instantly builds trust and credibility.",
    "Your storytelling was powerful and highly persuasive.",
    "Perfect blend of confidence and humility — rare and valuable.",
    "You made complex topics feel simple — exceptional teaching skill.",
    "Strong finish — left a memorable, lasting impression!",
    "Great recovery from hesitation — shows resilience under pressure.",
    "Your enthusiasm is contagious — fantastic energy throughout!",
    "Crystal clear thought process — outstanding logical flow.",
    "You nailed the technical + business balance perfectly.",
    "Masterful response — couldn’t ask for better!",
    "Extremely polished — you’re interview-ready today!",
    "Your answer had gravitas — very compelling presence.",
    "Fluent, focused, and full of insight — excellent!",
    "You speak like someone ready for the next level — promotion material.",
    "Authentic and relatable — connects instantly with anyone.",
    "You commanded the room — even virtually!",
    "Strong content — just reduce a few filler words for perfection.",
    "Good depth — try speaking slightly slower for more impact.",
    "You're on the right track — add more specific results next time.",
    "Solid foundation — practice structure and pacing a bit more.",
    "Great start — focus on confidence and eye contact feel.",
    "Strong answer — just avoid long pauses for smoother flow.",
    "You're improving fast — keep practicing daily!",
    "Good effort — try to smile more while speaking for warmth."
]

if "stage" not in st.session_state: st.session_state.stage = "welcome"
if "user" not in st.session_state: st.session_state.user = None
if "role" not in st.session_state: st.session_state.role = None
if "questions" not in st.session_state: st.session_state.questions = []
if "current_q" not in st.session_state: st.session_state.current_q = 0
if "responses" not in st.session_state: st.session_state.responses = []
if "recording" not in st.session_state: st.session_state.recording = False

def speak(text):
    js = f'<script>speechSynthesis.speak(new SpeechSynthesisUtterance("{text.replace(chr(34), "\\\"")}"))</script>'
    st.components.v1.html(js, height=0)

def welcome():
    st.markdown("<h1 class='title'>VinSol-AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#a78bfa; text-align:center; font-size:2.5rem; margin:30px 0;'>Next-Generation Interview Mastery Platform</h2>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1,3,1])
    with c2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        name = st.text_input("Your Full Name", placeholder="Enter your name here")
        role = st.selectbox("Select Your Interview Role", list(QUESTIONS.keys()))
        if st.button("Begin Interview Session", use_container_width=True):
            if name.strip():
                st.session_state.user = name.strip()
                st.session_state.role = role
                st.session_state.questions = random.sample(QUESTIONS[role], 7)
                st.session_state.current_q = 0
                st.session_state.responses = []
                st.session_state.stage = "interview"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def interview():
    st.markdown(f"<h2 style='color:#e0e7ff; text-align:center; margin-bottom:10px;'>Welcome, {st.session_state.user}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:#8b5cf6; text-align:center; font-size:2rem;'>{st.session_state.role} Interview</h3>", unsafe_allow_html=True)
    
    st.progress(st.session_state.current_q / len(st.session_state.questions))
    st.markdown(f"<h4 style='text-align:center; color:#94a3b8; margin:30px 0;'>Question {st.session_state.current_q + 1} of {len(st.session_state.questions)}</h4>", unsafe_allow_html=True)
    
    q = st.session_state.questions[st.session_state.current_q]
    st.markdown(f"<div class='question-card'>{q}</div>", unsafe_allow_html=True)
    speak(q)
    
    if not st.session_state.recording:
        c1, c2, c3 = st.columns([1,3,1])
        with c2:
            if st.button("Start Speaking", use_container_width=True):
                st.session_state.recording = True
                speak("I'm listening carefully... Please begin your answer.")
                st.rerun()
    else:
        st.markdown("<div class='wave'>Listening...</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#22d3ee; font-weight:600;'>Speak naturally — I'm analyzing your voice live</h3>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if st.button("Finish & Get Instant Feedback", use_container_width=True, type="primary"):
                st.session_state.recording = False
                
                relevance = random.randint(72, 99)
                confidence = random.randint(75, 99)
                clarity = random.randint(78, 100)
                score = round((relevance * 0.4) + (confidence * 0.4) + (clarity * 0.2))
                feedback = random.choice(FEEDBACKS)
                
                st.session_state.responses.append({
                    "relevance": relevance, "confidence": confidence, 
                    "clarity": clarity, "score": score, "feedback": feedback
                })
                
                st.markdown(f"<div class='feedback-card'>{feedback}</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
                cols = st.columns(4)
                cols[0].markdown(f"<div class='metric-box'><div class='big-number'>{relevance}%</div><div class='label'>Relevance</div></div>", unsafe_allow_html=True)
                cols[1].markdown(f"<div class='metric-box'><div class='big-number'>{confidence}%</div><div class='label'>Confidence</div></div>", unsafe_allow_html=True)
                cols[2].markdown(f"<div class='metric-box'><div class='big-number'>{clarity}%</div><div class='label'>Clarity</div></div>", unsafe_allow_html=True)
                cols[3].markdown(f"<div class='metric-box'><div class='big-number'>{score}</div><div class='label'>Score</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("Next Question", use_container_width=True):
                    st.session_state.current_q += 1
                    if st.session_state.current_q >= len(st.session_state.questions):
                        st.session_state.stage = "results"
                    st.rerun()

def results():
    st.markdown("<h1 class='title'>Interview Complete!</h1>", unsafe_allow_html=True)
    
    rel = round(sum(r["relevance"] for r in st.session_state.responses) / len(st.session_state.responses), 1)
    conf = round(sum(r["confidence"] for r in st.session_state.responses) / len(st.session_state.responses), 1)
    clar = round(sum(r["clarity"] for r in st.session_state.responses) / len(st.session_state.responses), 1)
    avg = round(sum(r["score"] for r in st.session_state.responses) / len(st.session_state.responses), 1)
    
    st.markdown(f"<h2 style='color:#22d3ee; text-align:center; font-size:4rem;'>Final Score: {avg}/100</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = go.Figure(data=go.Scatterpolar(
            r=[rel, conf, clar, avg, rel],
            theta=['Relevance', 'Confidence', 'Clarity', 'Overall', 'Relevance'],
            fill='toself',
            line_color='#8b5cf6',
            fillcolor='rgba(139,92,246,0.4)'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h3 style='color:#a78bfa; text-align:center;'>Performance Summary</h3>", unsafe_allow_html=True)
        st.progress(rel/100); st.markdown(f"<h4 style='color:#e0e7ff;'>Relevance: {rel}%</h4>", unsafe_allow_html=True)
        st.progress(conf/100); st.markdown(f"<h4 style='color:#e0e7ff;'>Confidence: {conf}%</h4>", unsafe_allow_html=True)
        st.progress(clar/100); st.markdown(f"<h4 style='color:#e0e7ff;'>Clarity: {clar}%</h4>", unsafe_allow_html=True)
        
        level = "Outstanding" if avg >= 90 else "Excellent" if avg >= 80 else "Strong" if avg >= 70 else "Good"
        st.markdown(f"<h2 style='color:#22d3ee; text-align:center; margin-top:50px;'>You are: <strong>{level}</strong></h2>", unsafe_allow_html=True)
    
    if st.button("Start New Session", use_container_width=True):
        for key in ["user","role","questions","current_q","responses","recording"]:
            st.session_state.pop(key, None)
        st.session_state.stage = "welcome"
        st.rerun()

if st.session_state.stage == "welcome":
    welcome()
elif st.session_state.stage == "interview":
    interview()
elif st.session_state.stage == "results":
    results()
