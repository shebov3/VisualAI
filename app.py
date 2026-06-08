"""
Real-Time Visual Assistant - Flask Web App
==========================================
Uses the Gemini Live API with camera + audio input from the browser.
No PyAudio needed — the browser handles all media capture.

Setup:
    pip install flask flask-socketio google-genai

Run:
    python app.py
"""

import asyncio
import base64
import os
import threading
import traceback

from dotenv import load_dotenv
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it in a .env file or environment variables."
    )

MODEL = "models/gemini-3.1-flash-live-preview"

SYSTEM_PROMPT = """You are a real-time visual assistant for the user. Your job is to look at the camera feed and clearly describe everything you see to support them in their daily life.

Follow these rules when describing the video or image:

1. Be Clear and Direct: Describe the scene in simple, plain English. Start with the most important things in the center of the frame or closest to the user.
2. Space and Location: Tell the user where objects are located in relation to them (e.g., "There is a coffee cup about six inches to your right," or "A doorway is straight ahead").
3. Read Text Aloud: If you see signs, labels, computer screens, or documents, read the text clearly. State what the object is before reading the text (e.g., "The label on the medicine bottle says...").
4. Identify Hazards: Immediately warn the user about potential dangers, such as obstacles on the floor, spills, stairs, or oncoming traffic.
5. Keep it Concise: Give enough detail to be helpful, but do not overwhelm the user with too much talking. Focus on what matters most for safety and understanding.
6. Tone: Speak in a calm, helpful, and friendly voice."""

# ─── Flask App ───────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "visual-assistant-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Gemini client
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=API_KEY,
)

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction=types.Content(
        parts=[types.Part(text=SYSTEM_PROMPT)]
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25000,
        sliding_window=types.SlidingWindow(target_tokens=12500),
    ),
)

# Store per-client sessions
sessions = {}


def run_gemini_session(sid):
    """Run the Gemini Live session in its own asyncio event loop (in a thread)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    state = sessions.get(sid)
    if state:
        state["loop"] = loop

    async def _session_loop():
        try:
            async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
                if not state:
                    return
                state["session"] = session
                state["ready"] = True
                socketio.emit("status", {"message": "Connected to Gemini", "connected": True}, to=sid)
                print(f"[{sid}] Gemini session connected")

                # Receive loop — get audio back from Gemini and send to browser
                while state.get("active"):
                    try:
                        turn = session.receive()
                        async for response in turn:
                            if data := response.data:
                                # Send raw PCM audio back to the browser
                                audio_b64 = base64.b64encode(data).decode("utf-8")
                                socketio.emit("audio_response", {"audio": audio_b64}, to=sid)
                            if text := response.text:
                                socketio.emit("text_response", {"text": text}, to=sid)
                    except Exception as e:
                        if state.get("active"):
                            print(f"[{sid}] Receive error: {e}")
                            break

        except Exception as e:
            traceback.print_exc()
            socketio.emit("status", {"message": f"Error: {e}", "connected": False}, to=sid)
        finally:
            if sid in sessions:
                sessions[sid]["active"] = False
                sessions[sid]["ready"] = False
            socketio.emit("status", {"message": "Disconnected from Gemini", "connected": False}, to=sid)
            print(f"[{sid}] Gemini session ended")

    loop.run_until_complete(_session_loop())
    loop.close()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── Socket.IO Events ───────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    sid = __import__("flask").request.sid
    print(f"[{sid}] Client connected")
    emit("status", {"message": "Connected to server. Click Start to begin.", "connected": False})


@socketio.on("disconnect")
def handle_disconnect():
    sid = __import__("flask").request.sid
    print(f"[{sid}] Client disconnected")
    if sid in sessions:
        sessions[sid]["active"] = False
        del sessions[sid]


@socketio.on("start_session")
def handle_start_session():
    sid = __import__("flask").request.sid
    print(f"[{sid}] Starting Gemini session...")

    if sid in sessions and sessions[sid].get("active"):
        emit("status", {"message": "Session already active", "connected": True})
        return

    sessions[sid] = {
        "session": None,
        "active": True,
        "ready": False,
        "loop": None,
    }

    # Launch the Gemini session in a background thread
    thread = threading.Thread(target=run_gemini_session, args=(sid,), daemon=True)
    thread.start()

    emit("status", {"message": "Connecting to Gemini...", "connected": False})


@socketio.on("stop_session")
def handle_stop_session():
    sid = __import__("flask").request.sid
    print(f"[{sid}] Stopping session...")
    if sid in sessions:
        sessions[sid]["active"] = False


@socketio.on("audio_input")
def handle_audio_input(data):
    """Receive audio chunks from the browser microphone."""
    sid = __import__("flask").request.sid
    state = sessions.get(sid)
    if not state or not state.get("ready") or not state.get("session") or not state.get("loop"):
        return

    audio_b64 = data.get("audio", "")
    if not audio_b64:
        return

    audio_bytes = base64.b64decode(audio_b64)
    session = state["session"]
    loop = state["loop"]

    # Schedule the send_realtime_input coroutine thread-safely on the loop
    coro = session.send_realtime_input(audio={"data": audio_bytes, "mime_type": "audio/pcm"})
    asyncio.run_coroutine_threadsafe(coro, loop)


@socketio.on("video_frame")
def handle_video_frame(data):
    """Receive a video frame from the browser camera."""
    sid = __import__("flask").request.sid
    state = sessions.get(sid)
    if not state or not state.get("ready") or not state.get("session") or not state.get("loop"):
        return

    frame_b64 = data.get("frame", "")
    if not frame_b64:
        return

    # Strip data URL prefix if present
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",")[1]

    frame_bytes = base64.b64decode(frame_b64)
    session = state["session"]
    loop = state["loop"]

    coro = session.send_realtime_input(video={"data": frame_bytes, "mime_type": "image/jpeg"})
    asyncio.run_coroutine_threadsafe(coro, loop)


@socketio.on("text_input")
def handle_text_input(data):
    """Receive text input from the browser."""
    sid = __import__("flask").request.sid
    state = sessions.get(sid)
    if not state or not state.get("ready") or not state.get("session") or not state.get("loop"):
        return

    text = data.get("text", "").strip()
    if not text:
        return

    session = state["session"]
    loop = state["loop"]

    coro = session.send_realtime_input(text=text)
    asyncio.run_coroutine_threadsafe(coro, loop)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Real-Time Visual Assistant")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
