import time
import cv2
import whisper
import numpy as np
import sounddevice as sd
import librosa
from queue import Queue

from utils.video_utils import start_webcam_capture, capture_frame, release_capture
from utils.captioning_utils import load_blip_model, generate_caption
from config.settings import CAPTION_FILE_PATH, WEBCAM_INDEX, MAX_TOKENS


whisper_model = whisper.load_model("small")   
processor, blip_model, device = load_blip_model()


cap = start_webcam_capture(WEBCAM_INDEX)


SR = 16000
CHANNELS = 1
BLOCK = 1024
SILENCE_THRESH = 0.01    
TRANSCRIBE_INTERVAL = 3  
last_trans_time = 0
last_speech_cap = ""

def capture_audio(duration=1):
    q = Queue()
    def cb(indata, frames, t, status):
        if status:
            print("[AUDIO STATUS]", status)
        q.put(indata.copy())
    with sd.InputStream(samplerate=SR, channels=CHANNELS,
                        blocksize=BLOCK, callback=cb):
        frames = [q.get() for _ in range(int(SR/BLOCK*duration))]
    audio = np.concatenate(frames, axis=0).flatten().astype(np.float32)
    if SR != 16000:
        audio = librosa.resample(audio, orig_sr=SR, target_sr=16000)
    return audio

def should_transcribe(audio):
    rms = np.sqrt(np.mean(audio**2))
    return rms > SILENCE_THRESH

def transcribe_audio(audio):
    try:

        res = whisper_model.transcribe(audio, language="en")
        return res.get("text", "").strip()
    except Exception as e:
        print(f"[WHISPER ERROR] {e}")
        return ""

def main():
    global last_trans_time, last_speech_cap
    with open(CAPTION_FILE_PATH, "w", encoding="utf-8") as fout:
        print("Press 'q' to quit...")
        last_vis_cap = ""
        while True:
            frame, ok = capture_frame(cap, skip=0)
            if not ok:
                break

            
            vis_cap = generate_caption(frame, processor, blip_model, device,
                                       max_tokens=MAX_TOKENS, last_caption=last_vis_cap)
            last_vis_cap = vis_cap

            
            now = time.time()
            speech_cap = ""
            if now - last_trans_time > TRANSCRIBE_INTERVAL:
                audio = capture_audio(duration=1)
                if should_transcribe(audio):
                    speech_cap = transcribe_audio(audio)
                    last_speech_cap = speech_cap or last_speech_cap
                last_trans_time = now

            
            parts = [vis_cap]
            if last_speech_cap:
                parts.append(last_speech_cap)
            full_cap = " | ".join(parts)

            
            cv2.putText(frame, full_cap, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Real-Time Captioning", frame)

            
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            fout.write(f"[{ts}] {full_cap}\n")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    release_capture(cap)

if __name__ == "__main__":
    main()
