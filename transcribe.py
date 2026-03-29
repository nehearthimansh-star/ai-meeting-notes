import whisper
import sys

file_path = sys.argv[1]

model = whisper.load_model("tiny")
result = model.transcribe(file_path)

print(result["text"])