# AI Meeting Notes Generator

This project is a web application that converts meeting audio into structured notes using Whisper for transcription and Hugging Face for note generation.

## Features

- Upload audio files such as `.mp3`, `.ogg`, `.wav`, and `.webm`
- Transcribe speech locally with Whisper
- Generate summaries, key points, and action items
- Use a simple browser-based interface

## Tech Stack

- Node.js
- Express
- Python
- Whisper
- Hugging Face API
- HTML, CSS, JavaScript
- FFmpeg

## How It Works

1. A user uploads an audio file.
2. The server sends the file to Whisper through `transcribe.py`.
3. Whisper returns transcript text.
4. The transcript is sent to Hugging Face for structured notes.
5. The UI shows the transcript, summary, key points, and action items.

## Local Run

1. Copy `.env.example` to `.env`.
2. Set `HF_API_KEY`.
3. Run `npm start`.
4. Open `http://localhost:3000`.

## Deployment

This app is prepared for Docker-based deployment because it needs Node.js, Python, and `ffmpeg` together.

Deployment files included:

- `Dockerfile`
- `.dockerignore`
- `requirements.txt`
- `render.yaml`

### Render

1. Push this repo to GitHub.
2. Create a new Blueprint or Web Service in Render from the repo.
3. Render will detect `render.yaml` or the `Dockerfile`.
4. Add `HF_API_KEY` as an environment variable.
5. Deploy.

Health check endpoint:

- `/health`

### Railway or Any Docker Host

1. Create a project from this repo.
2. Deploy using the included `Dockerfile`.
3. Add `HF_API_KEY` as an environment variable.

## Notes

- Uploaded files are stored in the container filesystem, so they are temporary on most hosts.
- The server reads `PORT` from the environment for cloud deployment.
- The container uses `python3` internally through `PYTHON_BIN`.

## Future Improvements

- Real-time transcription
- Speaker identification
- Download notes as PDF
- More persistent file storage for uploaded audio

## Authors

- Himanshu Mali
- Nikunj Sen
