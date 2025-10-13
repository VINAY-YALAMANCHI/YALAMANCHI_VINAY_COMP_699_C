# AI Interview Simulator – Package Diagram and Architecture

## Overview

This document outlines the modular implementation of the **AI Interview Simulator** using a package-based architecture. Each package handles a specific group of responsibilities to ensure separation of concerns, maintainability, and scalability.

---

## Package Structure

| Package              | Responsibility |
|----------------------|----------------|
| `ui`                 | Manages the user interface: role selection, display of questions, transcripts, and feedback. |
| `interview_session`  | Controls the overall interview flow: adaptive question logic, timing, and user progression. |
| `audio`              | Handles microphone input, recording, and playback of candidate responses. |
| `speech_to_text`     | Converts audio responses into text using speech recognition services. |
| `text_to_speech`     | Converts text questions into speech to simulate a realistic interview scenario. |
| `nlp_analysis`       | Performs keyword matching and semantic similarity scoring using NLP models. |
| `feedback`           | Aggregates content and delivery feedback and generates recommendations. |
| `question_bank`      | Stores and manages interview questions, difficulty levels, keywords, and model answers. |
| `scoring`            | Calculates final scores, evaluates delivery metrics like filler words and fluency. |
| `export`             | Generates reports (PDF, CSV) containing audio links, scores, and feedback summaries. |
| `security`           | Manages consent logging, encryption, and secure data storage. |
| `logging`            | Tracks exceptions, system issues, and activity logs for auditing and debugging. |

---

## Package Diagram Visualization

Below is the visual representation of the package diagram:

![Package Diagram](./Package%20Diagram.jpg)

