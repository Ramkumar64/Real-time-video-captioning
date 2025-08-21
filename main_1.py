
import time
import cv2
import whisper

from utils.video_utils import start_webcam_capture, capture_frame, release_capture
from utils.captioning_utils_1 import (
    load_fast_image_captioner,
    fast_generate_caption,
)
from utils.audio_utils import capture_audio

from config.settings import CAPTION_FILE_PATH, WEBCAM_INDEX


img_model, img_feat, img_tokenizer = load_fast_image_captioner()
speech_model = whisper.load_model("tiny.en")  


cap = start_webcam_capture(WEBCAM_INDEX)

def transcribe_if_voice(audio, thresh=0.01):
    """
    Returns transcription if RMS energy > thresh, else empty string.
    """
    rms = (audio**2).mean() ** 0.5
    if rms < thresh:
        return ""
    try:
        result = speech_model.transcribe(audio, fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        print("[WHISPER ERROR]", e)
        return ""

def main():
    with open(CAPTION_FILE_PATH, "w", encoding="utf-8") as fout:
        print("Press 'q' to quit...")
        last_trans = 0

        while True:
            frame, ok = capture_frame(cap)
            if not ok:
                break

            
            vis_cap = fast_generate_caption(frame, img_model, img_feat, img_tokenizer)

            
            now = time.time()
            speech_cap = ""
            if now - last_trans > 1.5:
                audio = capture_audio(duration=0.5)
                speech_cap = transcribe_if_voice(audio)
                last_trans = now if speech_cap else last_trans

            
            parts = [vis_cap]
            if speech_cap:
                parts.append(speech_cap)
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
