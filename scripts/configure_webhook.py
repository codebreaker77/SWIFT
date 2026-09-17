"""
SWIFT Webhook Configurator
Automatically points the SignalWire/Twilio phone number's voice webhook
to the SWIFT server's /twilio/inbound endpoint.

Run this after starting ngrok and setting SWIFT_PUBLIC_URL in .env:
    python scripts/configure_webhook.py
"""

import os, sys
from dotenv import load_dotenv

load_dotenv()

PROVIDER    = os.getenv("PROVIDER", "signalwire").lower()
PUBLIC_URL  = os.getenv("SWIFT_PUBLIC_URL", "").rstrip("/")

if "YOUR-NGROK" in PUBLIC_URL:
    print("[ERROR] SWIFT_PUBLIC_URL is still the placeholder.")
    print("        Run 'ngrok http 8001', copy the https URL, and set it in .env")
    sys.exit(1)

endpoint_prefix = "signalwire" if PROVIDER == "signalwire" else "twilio"
webhook_url = f"{PUBLIC_URL}/{endpoint_prefix}/inbound"
status_url  = f"{PUBLIC_URL}/{endpoint_prefix}/status_callback"

if PROVIDER == "signalwire":
    from signalwire.rest import Client
    client = Client(
        os.getenv("SIGNALWIRE_PROJECT_ID"),
        os.getenv("SIGNALWIRE_API_TOKEN"),
        signalwire_space_url=os.getenv("SIGNALWIRE_SPACE"),
    )
    phone_number = os.getenv("SIGNALWIRE_PHONE_NUMBER")
    numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
    if not numbers:
        print(f"[ERROR] Number {phone_number} not found on this SignalWire account.")
        sys.exit(1)
    n = numbers[0]
    n.update(
        voice_url=webhook_url,
        voice_method="POST",
        status_callback=status_url,
        status_callback_method="POST",
    )
    print(f"[SWIFT] SignalWire webhook configured!")
    print(f"  Number  : {phone_number}")
    print(f"  Voice   : {webhook_url}")
else:
    from twilio.rest import Client
    client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    phone_number = os.getenv("TWILIO_PHONE_NUMBER")
    numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
    if not numbers:
        print(f"[ERROR] Number {phone_number} not found on this Twilio account.")
        sys.exit(1)
    n = numbers[0]
    n.update(voice_url=webhook_url, voice_method="POST")
    print(f"[SWIFT] Twilio webhook configured!")
    print(f"  Number  : {phone_number}")
    print(f"  Voice   : {webhook_url}")

print()
print(f"[SWIFT] The pipeline is now live.")
print(f"        Dial {phone_number} from any phone to start streaming.")
print(f"        Watch live sessions at: {PUBLIC_URL}/sessions")
