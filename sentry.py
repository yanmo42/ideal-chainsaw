import cv2
import time
import datetime
import requests
import os
import threading
import pyaudio
import wave

# --- CONFIG ---
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1452878573030342657/gEuqk53Vn8_53jKBaviJlBFE6UeOzExeBniWlsbZNhPzeqH1CSdL7nPY3K9GkACHPpEN"
MOTION_THRESHOLD = 1500
BUFFER_SECONDS = 5

def send_to_discord(filename, content="Alert"):
    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f)}
            requests.post(DISCORD_WEBHOOK_URL, data={'content': content}, files=files)
        os.remove(filename) 
    except: pass

def record_audio(filename, duration):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    frames = []
    for _ in range(0, int(fs / chunk * duration)):
        frames.append(stream.read(chunk))
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

def main():
    cap = cv2.VideoCapture(0)
    recording = False
    _, frame1 = cap.read()
    _, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion = any(cv2.contourArea(c) > MOTION_THRESHOLD for c in contours)

        if motion and not recording:
            recording = True
            # 1. IMMEDIATE SNAPSHOT (The Evidence)
            snap_name = "FACE_DETECTED.jpg"
            cv2.imwrite(snap_name, frame2)
            threading.Thread(target=send_to_discord, args=(snap_name, "⚠️ **INTRUDER ALERT: Snapshot sent!**")).start()

            # 2. START AUDIO & VIDEO
            # (In this simplified DIY version, we capture a set 10-second clip of both)
            audio_name = "audio_clip.wav"
            threading.Thread(target=record_audio, args=(audio_name, 10)).start()
            # Note: We'll send the video/audio once that 10s is up
            
        frame1 = frame2
        ret, frame2 = cap.read()
        if cv2.waitKey(1) == ord('q'): break

    cap.release()

if __name__ == "__main__":
    main()