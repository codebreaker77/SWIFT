"""
SWIFT Session Manager
Manages concurrent live Twilio call sessions, each with their own AudioRingBuffer,
inference state, and connected mobile telemetry WebSocket.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Any
from fastapi import WebSocket

from src.realtime_stream import AudioRingBuffer

# SPI thresholds from environment (loaded by twilio_server.py)
SPI_THREAT_THRESHOLD = 0.70
SPI_ELEVATED_THRESHOLD = 0.30


@dataclass
class CallSession:
    """Represents a single active call being monitored by SWIFT."""
    session_id: str
    call_sid: str = ""               # Assigned by Twilio when the stream connects
    stream_sid: str = ""             # Twilio Stream SID

    # Rolling audio buffer — 2 second window at 16kHz
    ring_buffer: AudioRingBuffer = field(default_factory=AudioRingBuffer)

    # Current inference state
    current_spi: float = 0.0
    status: Literal["waiting", "sampling", "authentic", "elevated", "threat"] = "waiting"

    # Metrics for telemetry payload
    last_metrics: Dict[str, float] = field(default_factory=dict)

    # Mobile app telemetry WebSocket
    client_ws: Optional[WebSocket] = None

    # Twilio stream WebSocket (server-side handle)
    twilio_ws: Optional[Any] = None

    # Timing
    created_at: float = field(default_factory=time.time)
    last_inference_at: float = 0.0
    sample_count: int = 0          # Number of 16kHz samples received


class SessionManager:
    """Thread-safe, asyncio-compatible in-memory store for active call sessions."""

    def __init__(self):
        self._sessions: Dict[str, CallSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str) -> CallSession:
        async with self._lock:
            session = CallSession(session_id=session_id)
            self._sessions[session_id] = session
            return session

    async def get(self, session_id: str) -> Optional[CallSession]:
        return self._sessions.get(session_id)

    async def get_by_call_sid(self, call_sid: str) -> Optional[CallSession]:
        for s in self._sessions.values():
            if s.call_sid == call_sid:
                return s
        return None

    async def remove(self, session_id: str):
        async with self._lock:
            self._sessions.pop(session_id, None)

    def all_sessions(self) -> Dict[str, CallSession]:
        return dict(self._sessions)

    async def broadcast_verdict(self, session: CallSession):
        """Push the latest SPI payload to the mobile app WebSocket."""
        payload = {
            "session_id": session.session_id,
            "call_sid": session.call_sid,
            "spi": round(session.current_spi, 4),
            "status": session.status,
            "metrics": session.last_metrics,
            "elapsed_ms": int((time.time() - session.created_at) * 1000),
        }
        if session.client_ws is not None:
            try:
                await session.client_ws.send_json(payload)
            except Exception:
                session.client_ws = None
        # Also broadcast to any general mobile listeners
        await self.broadcast_to_all(payload)

    async def broadcast_to_all(self, payload: dict):
        """Broadcast payload to all connected mobile clients."""
        for session in list(self._sessions.values()):
            if session.client_ws is not None:
                try:
                    await session.client_ws.send_json(payload)
                except Exception:
                    session.client_ws = None



# Global singleton used by twilio_server.py
session_manager = SessionManager()
