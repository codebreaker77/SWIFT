"""
SWIFT Telephony Pipeline Test Script
Dials YOUR_PHONE_NUMBER from the SWIFT number using the configured provider.
Run AFTER: uvicorn twilio_server:app --port 8001 + ngrok http 8001

Usage:
    python scripts/test_twilio_call.py
"""

import os, sys
from dotenv import load_dotenv

load_dotenv()

PROVIDER        = os.getenv("PROVIDER", "signalwire").lower()
PUBLIC_URL      = os.getenv("SWIFT_PUBLIC_URL", "").rstrip("/")
TO_NUMBER       = os.getenv("YOUR_PHONE_NUMBER")

if "YOUR-NGROK" in PUBLIC_URL:
    print("[ERROR] SWIFT_PUBLIC_URL is still the placeholder.")
    print("        Run 'ngrok http 8001' and set the URL in .env")
    sys.exit(1)

webhook_url = f"{PUBLIC_URL}/twilio/inbound"

if PROVIDER == "signalwire":
    from signalwire.rest import Client
    client = Client(
        os.getenv("SIGNALWIRE_PROJECT_ID"),
        os.getenv("SIGNALWIRE_API_TOKEN"),
        signalwire_space_url=os.getenv("SIGNALWIRE_SPACE"),
    )
    from_number = os.getenv("SIGNALWIRE_PHONE_NUMBER")
else:
    from twilio.rest import Client
    client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

print(f"[SWIFT] Provider  : {PROVIDER.upper()}")
print(f"[SWIFT] From      : {from_number}")
print(f"[SWIFT] To        : {TO_NUMBER}")
print(f"[SWIFT] Webhook   : {webhook_url}")
print()

call = client.calls.create(
    to=TO_NUMBER,
    from_=from_number,
    url=webhook_url,
    method="POST",
)

print(f"[SWIFT] Call initiated!")
print(f"  CallSid : {call.sid}")
print(f"  Status  : {call.status}")
print()
print(f"[SWIFT] Answer your phone. Watch SPI scores at: {PUBLIC_URL}/sessions")
print(f"[SWIFT] Live verdict at               : {PUBLIC_URL}/twilio/verdict/{call.sid}")
