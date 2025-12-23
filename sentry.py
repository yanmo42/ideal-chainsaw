import cv2
import time
import datetime
import requests
import os
import secrets
# --- CONFIGURATION ---
DISCORD_WEBHOOK_URL = secrets.DISCORD_WEBHOOK


MOTION_THRESHOLD = 1000
RECORDING_BUFFER_SECONDS = 5

# Optional: keep recordings in a folder
OUTPUT_DIR = "recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _post_to_discord(files=None, content=""):
    """Low-level helper. Returns True on success."""
    try:
        data = {"content": content} if content else {}
        resp = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files, timeout=20)
        return resp.status_code in (200, 204), resp
    except Exception as e:
        return False, e

def send_snapshot_to_discord(frame_bgr, label=""):
    """Encodes a frame to JPEG in-memory and uploads it immediately."""
    ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        print("Snapshot encode failed.")
        return False

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"🚨 **Security Alert**\nMotion detected at: {ts}"
    if label:
        content += f"\n{label}"

    files = {"file": ("snapshot.jpg", jpg.tobytes(), "image/jpeg")}
    success, resp = _post_to_discord(files=files, content=content)
    if success:
        print("Snapshot upload successful.")
    else:
        print(f"Snapshot upload failed: {resp}")
    return success

def send_video_to_discord(filepath, delete_on_success=True, retries=2, retry_delay=2.0):
    """Uploads the recorded video to Discord; deletes local file on success if requested."""
    print(f"Uploading {filepath} to Discord...")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"📹 **Motion Clip**\nCaptured at: {ts}\nFile: `{os.path.basename(filepath)}`"

    for attempt in range(retries + 1):
        try:
            with open(filepath, "rb") as f:
                files = {"file": (os.path.basename(filepath), f, "video/mp4")}
                success, resp = _post_to_discord(files=files, content=content)

            if success:
                print("Video upload successful.")
                if delete_on_success:
                    try:
                        os.remove(filepath)
                        print("Local video deleted to save space.")
                    except Exception as e:
                        print(f"Warning: could not delete {filepath}: {e}")
                return True

            print(f"Video upload failed (attempt {attempt+1}/{retries+1}): {resp}")
        except Exception as e:
            print(f"Video send error (attempt {attempt+1}/{retries+1}): {e}")

        if attempt < retries:
            time.sleep(retry_delay)

    return False

def main():
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)

    recording = False
    last_motion_time = 0
    out = None
    filename = None

    # Read initial frames
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    if not (ret1 and ret2):
        raise RuntimeError("Could not read from webcam. Check permissions / camera availability.")

    print("Sentry System Active. Press 'q' to quit.")
    snapshot_sent_for_this_event = False

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < MOTION_THRESHOLD:
                continue
            motion_detected = True
            last_motion_time = time.time()

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # --- Instant evidence on first motion frame ---
        if motion_detected and not snapshot_sent_for_this_event:
            # Send a quick snapshot ASAP (doesn't touch disk)
            send_snapshot_to_discord(frame1, label="(Instant snapshot)")
            snapshot_sent_for_this_event = True

        # --- Start recording on first motion ---
        if motion_detected and not recording:
            recording = True
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(OUTPUT_DIR, f"motion_{timestamp}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            print(f"Started recording: {filename}")

        # --- Recording loop ---
        if recording and out is not None:
            out.write(frame1)

            # Stop if no motion for buffer seconds
            if time.time() - last_motion_time > RECORDING_BUFFER_SECONDS:
                recording = False
                out.release()
                out = None
                print("Motion stopped. Saving and uploading video...")
                send_video_to_discord(filename, delete_on_success=True)

                # reset event flags
                filename = None
                snapshot_sent_for_this_event = False

        cv2.imshow("Sentry View", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            print("Frame read failed; ending.")
            break

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
