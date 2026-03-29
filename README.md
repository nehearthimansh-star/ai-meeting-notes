# 🤖 AI Meeting Notes Generator

## 📌 Project Overview
This project is an AI-powered web application that converts meeting audio into structured notes.  
It uses speech recognition and natural language processing to generate summaries, key points, and action items automatically.

---

## 🚀 Features
- 🎤 Upload audio files (.mp3, .ogg, .wav, .webm)
- 🧠 Speech-to-text using Whisper (local)
- 🤖 AI-based summarization using Hugging Face
- 📌 Extracts:
  - Summary
  - Key Points
  - Action Items
- 🌐 Simple and interactive web interface

---

## 🛠️ Technologies Used
- Node.js (Backend)
- Express.js
- Python (Whisper)
- Hugging Face API
- HTML, CSS, JavaScript
- FFmpeg

---

## ⚙️ How It Works

1. User uploads audio file  
2. Backend sends audio to Whisper (Python)  
3. Whisper converts speech → text  
4. Text is sent to Hugging Face model  
5. AI generates structured meeting notes  
6. Output is displayed on the website  

---


---

## 📷 Output Example
- Transcript of meeting
- AI-generated summary
- Key points and action items

---

## ⚠️ Challenges Faced
- Setting up FFmpeg for Whisper  
- Managing API limits  
- Handling large audio files  
- Integrating Node.js with Python  

---

## 🔮 Future Improvements
- Real-time transcription  
- Speaker identification  
- Download notes as PDF  
- Cloud deployment  

---

## 🧠 Key Learning
- Speech recognition using Whisper  
- Prompt engineering for better summaries  
- Backend integration with AI models  
- Secure API handling using `.gitignore`  

---

## 👨‍💻 Author
- Himanshu Mali
- Nikunj Sen 

---

## 🙌 Acknowledgement
This project was developed as part of academic coursework to demonstrate real-world AI applications.

---
