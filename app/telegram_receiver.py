import atexit
import os
import logging
import asyncio
import json
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from telegram.constants import ChatAction

import fcntl
import signal
import sys

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("telegram_receiver")

LOCK_FILE = "/tmp/telegram_receiver.lock"

def _cleanup_lock():
    """Remove lock file on exit."""
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass

try:
    lock_fp = open(LOCK_FILE, "w")
    fcntl.lockf(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_fp.write(str(os.getpid()))
    lock_fp.flush()
    atexit.register(_cleanup_lock)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # Ensure atexit runs on SIGTERM
except BlockingIOError:
    logger.error("Another instance of Telegram Receiver is already running. Exiting.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Failed to acquire lock: {e}")
    sys.exit(1)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:9000/chat")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set in the environment variables.")
    exit(1)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = str(update.message.chat_id)
    text = update.message.text
    
    logger.info(f"Received message from {chat_id}: {text}")

    try:
        # Show typing action while the gateway processes the request
        await context.bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.TYPING)

        # Forward the message to the Agent Hub Gateway
        async with httpx.AsyncClient(timeout=120) as client:
            payload = {
                "prompt": text,
                "session_id": chat_id,
                "context": {
                    "source": "telegram",
                    "user_id": update.message.from_user.id,
                    "username": update.message.from_user.username
                }
            }
            response = await client.post(GATEWAY_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            reply_text = data.get("output", "No response received from Gateway.")
            
            # Send the response back to Telegram
            await update.message.reply_text(reply_text)
            
            # Additional logic: Process exact reports bypassing the LLM
            reports = data.get("reports")
            if reports:
                for report in reports:
                    MAX_LEN = 4000
                    chunks = []
                    current_chunk = ""
                    for par in report.split('\n\n'):
                        if len(current_chunk) + len(par) + 2 > MAX_LEN:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = par
                        else:
                            current_chunk += ("\n\n" + par) if current_chunk else par
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        
                    for chunk in chunks:
                        if not chunk: continue
                        # Safety fallback for individual massive paragraphs
                        while len(chunk) > MAX_LEN:
                            sub_chunk = chunk[:MAX_LEN]
                            await update.message.reply_text(sub_chunk)
                            chunk = chunk[MAX_LEN:]
                        if chunk:
                            await update.message.reply_text(chunk)
            
    except Exception as e:
        logger.error(f"Error forwarding message to Hub Gateway: {e}")
        await update.message.reply_text("⚠️ Sorry, I couldn't reach the Hub Gateway right now.")

if __name__ == "__main__":
    logger.info("Starting Telegram Bot Receiver...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Handle all text messages (including commands right now to forward everything to LLM)
    all_text_handler = MessageHandler(filters.TEXT, handle_message)
    app.add_handler(all_text_handler)
    
    logger.info("Listening for messages...")
    app.run_polling()
