import os
import asyncio
import base64
import json
import time
import numpy as np
import scipy.signal as signal
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ---- Load environment ----
load_dotenv()

PROVIDER = os.getenv("PROVIDER", "signalwire").lower()

# Twilio credentials
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# SignalWire credentials
SW_SPACE        = os.getenv("SIGNALWIRE_SPACE", "")
SW_PROJECT_ID   = os.getenv("SIGNALWIRE_PROJECT_ID", "")
SW_API_TOKEN    = os.getenv("SIGNALWIRE_API_TOKEN", "")
SW_PHONE_NUMBER = os.getenv("SIGNALWIRE_PHONE_NUMBER", "")

SWIFT_PUBLIC_URL = os.getenv("SWIFT_PUBLIC_URL", "https://YOUR-NGROK-URL.ngrok-free.app")

SPI_THREAT_THRESHOLD   = float(os.getenv("SPI_THREAT_THRESHOLD", "0.70"))
SPI_ELEVATED_THRESHOLD = float(os.getenv("SPI_ELEVATED_THRESHOLD", "0.30"))
INFERENCE_HOP_MS       = int(os.getenv("INFERENCE_HOP_MS", "500"))

# ---- Build telephony client (Twilio or SignalWire) ----
telephony_client = None
active_phone_number = ""

if PROVIDER == "signalwire" and SW_PROJECT_ID:
    from signalwire.rest import Client as SWClient
    telephony_client = SWClient(
        SW_PROJECT_ID,
        SW_API_TOKEN,
        signalwire_space_url=SW_SPACE,
    )
    active_phone_number = SW_PHONE_NUMBER
    print(f"[SWIFT] Provider: SignalWire | Number: {active_phone_number}")
elif PROVIDER == "twilio" and TWILIO_ACCOUNT_SID:
    from twilio.rest import Client as TwilioClient
    telephony_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    active_phone_number = TWILIO_PHONE_NUMBER
    print(f"[SWIFT] Provider: Twilio | Number: {active_phone_number}")
else:
    print(f"[SWIFT] WARNING: No telephony provider configured. Mitigation API will be disabled.")

# ---- Import SWIFT internals ----
from src.config import AudioConfig, LFCCConfig, ModelConfig, CHECKPOINT_DIR
from src.realtime_stream import RealtimeStreamDetector, AudioRingBuffer
from src.session_manager import session_manager, CallSession

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_CFG = AudioConfig()

print(f"[SWIFT] Initializing PhysioSpecNet on {DEVICE}...")
STREAM_DETECTOR = RealtimeStreamDetector(audio_cfg=AUDIO_CFG)
print("[SWIFT] Model ready.")

# ---- FastAPI App ----
app = FastAPI(title="SWIFT Twilio Inference Server", version="2.0.0")
app.mount("/public", StaticFiles(directory="public"), name="public")


# =============================================================================
# Utility: μ-law decode + resample
# =============================================================================

def decode_mulaw(mulaw_bytes: bytes) -> np.ndarray:
    """Decode G.711 μ-law 8-bit bytes to float32 PCM in range [-1.0, 1.0]."""
    data = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    data = ~data
    sign = data & 0x80
    exponent = (data & 0x70) >> 4
    mantissa = data & 0x0F
    sample = ((mantissa.astype(np.int32) << 3) + 132) << exponent
    sample = sample - 132
    sample = np.where(sign != 0, -sample, sample)
    return (sample / 32768.0).astype(np.float32)


def resample_8k_to_16k(audio_8k: np.ndarray) -> np.ndarray:
    """Upsample 8kHz telephony audio to 16kHz using the Fourier method."""
    if len(audio_8k) == 0:
        return np.zeros(0, dtype=np.float32)
    return signal.resample(audio_8k, len(audio_8k) * 2).astype(np.float32)


def determine_status(spi: float) -> str:
    if spi >= SPI_THREAT_THRESHOLD:
        return "threat"
    elif spi >= SPI_ELEVATED_THRESHOLD:
        return "elevated"
    else:
        return "authentic"


# =============================================================================
# Phase 1 — Twilio Webhook: TwiML Response
# POST /twilio/inbound
# Twilio calls this when the user's app dials the SWIFT Twilio number.
# We respond with TwiML to fork the audio stream to our WebSocket endpoint.
# =============================================================================

@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    """
    TwiML webhook — instructs Twilio to:
      1. Start a Media Stream (fork audio to our WebSocket)
      2. Hold the line open for 60 seconds while we analyze
    """
    form = await request.form()
    call_sid = form.get("CallSid", "")
    from_number = form.get("From", "")
    print(f"[SWIFT] Inbound call: CallSid={call_sid}, From={from_number}")

    stream_url = f"{SWIFT_PUBLIC_URL.replace('https://', 'wss://')}/twilio/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{stream_url}" track="inbound_track" />
    </Start>
    <Pause length="60"/>
</Response>"""

    return Response(content=twiml, media_type="text/xml")


# =============================================================================
# Phase 2 — Twilio Media Stream WebSocket
# WS /twilio/stream
# This is where Twilio connects and continuously sends base64 μ-law audio.
# =============================================================================

@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """
    Twilio Media Streams WebSocket handler.
    Decodes incoming μ-law audio, feeds into AudioRingBuffer, runs
    PhysioSpecNet inference every 500ms, and broadcasts SPI scores to
    the paired mobile app WebSocket.
    """
    await websocket.accept()
    print("[SWIFT] Twilio Media Stream WebSocket connected.")

    session: CallSession = None
    ring_buffer = AudioRingBuffer(capacity=AUDIO_CFG.num_samples)
    last_inference_time = 0.0
    hop_s = INFERENCE_HOP_MS / 1000.0

    try:
        async for raw_message in websocket.iter_text():
            msg = json.loads(raw_message)
            event = msg.get("event", "")

            # --- Twilio Connected event ---
            if event == "connected":
                print(f"[SWIFT] Twilio stream connected: protocol={msg.get('protocol')}")

            # --- Twilio Start event: gives us CallSid and StreamSid ---
            elif event == "start":
                start_data = msg.get("start", {})
                call_sid = start_data.get("callSid", "")
                stream_sid = start_data.get("streamSid", "")
                custom_params = start_data.get("customParameters", {})
                session_id = custom_params.get("session_id", call_sid)

                print(f"[SWIFT] Stream started: CallSid={call_sid}, StreamSid={stream_sid}, SessionId={session_id}")

                # Find or create the session (mobile app may have pre-registered it)
                session = await session_manager.get(session_id)
                if session is None:
                    session = await session_manager.create_session(session_id)

                session.call_sid = call_sid
                session.stream_sid = stream_sid
                session.twilio_ws = websocket
                session.status = "sampling"

                last_inference_time = time.time()

            # --- Twilio Media event: contains the actual audio ---
            elif event == "media":
                if session is None:
                    continue  # haven't received start event yet

                media = msg.get("media", {})
                payload_b64 = media.get("payload", "")

                if not payload_b64:
                    continue

                # Decode μ-law → float32 PCM, resample 8kHz → 16kHz
                mulaw_bytes = base64.b64decode(payload_b64)
                audio_8k = decode_mulaw(mulaw_bytes)
                audio_16k = resample_8k_to_16k(audio_8k)

                # Push into ring buffer
                ring_buffer.append(audio_16k)
                session.sample_count += len(audio_16k)

                # Run inference every hop_s seconds
                now = time.time()
                if (now - last_inference_time) >= hop_s:
                    last_inference_time = now

                    # VAD check — skip inference on silence
                    window = ring_buffer.get_window()
                    rms = float(np.sqrt(np.mean(window ** 2)))
                    peak = float(np.max(np.abs(window)))

                    if rms < 0.002 or peak < 0.025:
                        spi = 0.02
                    else:
                        result = STREAM_DETECTOR.process_chunk(audio_16k)
                        spi = float(result.get("smoothed_p_fake", 0.0))

                    session.current_spi = spi
                    session.status = determine_status(spi)
                    session.last_inference_at = now
                    session.last_metrics = {
                        "feature_extraction_ms": result.get("feature_extraction_ms", 0) if rms >= 0.002 else 0,
                        "inference_ms": result.get("inference_ms", 0) if rms >= 0.002 else 0,
                        "total_latency_ms": result.get("total_latency_ms", 0) if rms >= 0.002 else 0,
                    }

                    print(f"[SWIFT] SPI={spi:.3f} | Status={session.status} | RMS={rms:.4f}")

                    # Push result to paired mobile WebSocket
                    await session_manager.broadcast_verdict(session)

                    # Automated mitigation on THREAT
                    if session.status == "threat" and twilio_client and session.call_sid:
                        await _trigger_mitigation(session.call_sid, action="warn")

            # --- Twilio Stop event: call ended ---
            elif event == "stop":
                print(f"[SWIFT] Stream stopped: {msg.get('stop', {})}")
                break

    except WebSocketDisconnect:
        print("[SWIFT] Twilio WebSocket disconnected.")
    except Exception as e:
        print(f"[SWIFT] Error in Twilio stream handler: {e}")
    finally:
        if session:
            await session_manager.remove(session.session_id)
            print(f"[SWIFT] Session {session.session_id} cleaned up.")


# =============================================================================
# Phase 2 — Mobile App Telemetry WebSocket
# WS /swift/telemetry/{session_id}
# The Android app connects here to receive live SPI scores for the HUD.
# =============================================================================

@app.websocket("/swift/telemetry/{session_id}")
async def mobile_telemetry(websocket: WebSocket, session_id: str):
    """
    Mobile app telemetry channel.
    The Android app connects here immediately after pressing 'Scan'.
    This WebSocket receives live SPI scores pushed by the inference loop.
    """
    await websocket.accept()
    print(f"[SWIFT] Mobile telemetry connected for session: {session_id}")

    # Pre-create session so it's ready when Twilio stream arrives
    session = await session_manager.get(session_id)
    if session is None:
        session = await session_manager.create_session(session_id)

    # Register this WebSocket as the mobile app's telemetry channel
    session.client_ws = websocket

    try:
        # Keep the connection alive — inference loop pushes results to it
        # Client can send "disconnect" to clean up early
        async for raw_msg in websocket.iter_text():
            data = json.loads(raw_msg)
            if data.get("action") == "disconnect":
                break
            # Future: handle action commands from the app (e.g. sever call)
            elif data.get("action") == "sever":
                if session.call_sid and twilio_client:
                    await _trigger_mitigation(session.call_sid, action="sever")

    except WebSocketDisconnect:
        print(f"[SWIFT] Mobile telemetry disconnected for session: {session_id}")
    finally:
        if session:
            session.client_ws = None


# =============================================================================
# Phase 2 — REST Verdict Endpoint
# GET /twilio/verdict/{call_sid}
# Polling fallback for mobile apps that can't maintain a persistent WebSocket.
# =============================================================================

@app.get("/twilio/verdict/{call_sid}")
async def get_verdict(call_sid: str):
    """Return the latest SPI score and status for a given CallSid."""
    session = await session_manager.get_by_call_sid(call_sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found for this CallSid.")

    return {
        "session_id": session.session_id,
        "call_sid": call_sid,
        "spi": round(session.current_spi, 4),
        "status": session.status,
        "elapsed_ms": int((time.time() - session.created_at) * 1000),
        "metrics": session.last_metrics,
    }


# =============================================================================
# Phase 2 — Mitigation Trigger
# POST /twilio/action/{call_sid}
# Called by the mobile app to trigger warn or sever on demand.
# =============================================================================

@app.post("/twilio/action/{call_sid}")
async def trigger_action(call_sid: str, request: Request):
    """Trigger a Twilio REST API action on a live call (warn or sever)."""
    body = await request.json()
    action = body.get("action", "warn")  # "warn" or "sever"
    result = await _trigger_mitigation(call_sid, action=action)
    return result


async def _trigger_mitigation(call_sid: str, action: str = "warn") -> dict:
    """Execute REST API mitigation on an active call via the configured provider."""
    if not telephony_client:
        return {"status": "error", "detail": "No telephony provider configured. Check PROVIDER in .env."}

    try:
        if action == "warn":
            telephony_client.calls(call_sid).update(
                twiml='<Response><Say voice="Polly.Matthew">Warning. Synthetic voice detected on this call.</Say></Response>'
            )
            print(f"[SWIFT] Mitigation: Whisper warning injected into CallSid={call_sid}")
            return {"status": "ok", "action": "warn", "call_sid": call_sid}

        elif action == "sever":
            telephony_client.calls(call_sid).update(status="completed")
            print(f"[SWIFT] Mitigation: Call severed for CallSid={call_sid}")
            return {"status": "ok", "action": "sever", "call_sid": call_sid}

        else:
            return {"status": "error", "detail": f"Unknown action: {action}"}

    except Exception as e:
        print(f"[SWIFT] Mitigation error: {e}")
        return {"status": "error", "detail": str(e)}


# =============================================================================
# Health + Status
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "provider": PROVIDER,
        "active_number": active_phone_number,
        "telephony_configured": telephony_client is not None,
        "active_sessions": len(session_manager.all_sessions()),
        "public_url": SWIFT_PUBLIC_URL,
        "spi_thresholds": {
            "elevated": SPI_ELEVATED_THRESHOLD,
            "threat": SPI_THREAT_THRESHOLD,
        },
    }


@app.get("/sessions")
async def list_sessions():
    sessions = session_manager.all_sessions()
    return {
        "active_sessions": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "call_sid": s.call_sid,
                "status": s.status,
                "spi": round(s.current_spi, 4),
                "sample_count": s.sample_count,
                "elapsed_s": round(time.time() - s.created_at, 1),
                "mobile_connected": s.client_ws is not None,
            }
            for s in sessions.values()
        ],
    }

