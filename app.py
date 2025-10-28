import tkinter as tk
from tkinter import ttk, messagebox
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
import threading
import os
import tempfile
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

tts_engine = pyttsx3.init()
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class InterviewApp:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Interview Simulator")
        self.roles_data = {}
        self.keywords_data = {}
        self.role_var = tk.StringVar()
        self.diff_var = tk.StringVar()
        self.time_limit = tk.IntVar(value=30)
        self.current_question = ""
        self.recording = False
        self.setup_ui()

    def setup_ui(self):
        ttk.Label(self.master, text="Select Role:").pack()
        self.role_combo = ttk.Combobox(self.master, textvariable=self.role_var)
        self.role_combo.pack()
        ttk.Label(self.master, text="Select Difficulty:").pack()
        self.diff_combo = ttk.Combobox(self.master, textvariable=self.diff_var, values=["easy", "medium", "hard"])
        self.diff_combo.pack()
        ttk.Label(self.master, text="Time Limit (seconds):").pack()
        ttk.Spinbox(self.master, from_=10, to=180, textvariable=self.time_limit).pack()
        ttk.Button(self.master, text="Start Interview", command=self.start_interview).pack(pady=10)
        self.question_label = ttk.Label(self.master, text="", wraplength=400)
        self.question_label.pack(pady=10)
        self.transcript_text = tk.Text(self.master, height=5, width=50)
        self.transcript_text.pack(pady=10)
        self.feedback_label = ttk.Label(self.master, text="", wraplength=400)
        self.feedback_label.pack(pady=10)

    def load_data(self, roles_dict, keywords_dict):
        self.roles_data = roles_dict
        self.keywords_data = keywords_dict
        self.role_combo.config(values=list(roles_dict.keys()))

    def start_interview(self):
        role = self.role_var.get()
        diff = self.diff_var.get()
        if role not in self.roles_data or diff not in self.roles_data[role]:
            messagebox.showerror("Error", "Select valid role and difficulty.")
            return
        questions = self.roles_data[role][diff]
        if not questions:
            messagebox.showerror("Error", "No questions available for this selection.")
            return
        self.current_question = questions[0]
        self.question_label.config(text=self.current_question)
        threading.Thread(target=self.play_and_record, daemon=True).start()

    def play_and_record(self):
        tts_engine.say(self.current_question)
        tts_engine.runAndWait()
        messagebox.showinfo("Recording", "Recording will start now. Answer aloud.")
        fs = 16000
        duration = self.time_limit.get()
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        temp_wav = tempfile.mktemp(suffix=".wav")
        wav.write(temp_wav, fs, audio)
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcript = "[Could not understand audio]"
        except sr.RequestError:
            transcript = "[Speech recognition failed]"
        os.remove(temp_wav)
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(tk.END, transcript)
        self.provide_feedback(transcript)

    def provide_feedback(self, transcript):
        question = self.current_question
        key_terms = self.keywords_data.get(question, [])
        tokens = word_tokenize(transcript.lower())
        keyword_matches = [kw for kw in key_terms if kw in tokens]
        keyword_score = len(keyword_matches) / len(key_terms) if key_terms else 0
        question_embedding = semantic_model.encode(question, convert_to_tensor=True)
        answer_embedding = semantic_model.encode(transcript, convert_to_tensor=True)
        semantic_score = float(util.pytorch_cos_sim(question_embedding, answer_embedding)[0][0])
        combined_score = (0.5 * keyword_score + 0.5 * semantic_score) * 100
        feedback = f"Keyword Score: {keyword_score:.2f}\nSemantic Similarity: {semantic_score:.2f}\nFinal Combined Score: {combined_score:.2f}/100"
        self.feedback_label.config(text=feedback)

if __name__ == "__main__":
    root = tk.Tk()
    app = InterviewApp(root)
    root.mainloop()
