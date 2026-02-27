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

# Longer timeout for Rothchild analysis runs (can take 3-5 minutes)
GATEWAY_TIMEOUT_SECONDS = int(os.getenv("GATEWAY_TIMEOUT", "420"))  # 7 minutes

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set in the environment variables.")
    exit(1)


async def _send_typing_loop(context, chat_id, stop_event: asyncio.Event):
    """Keep sending 'typing' action every 4s until stop_event is set."""
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=4)
        except asyncio.TimeoutError:
            pass


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = str(update.message.chat_id)
    text = update.message.text

    logger.info(f"Received message from {chat_id}: {text[:80]}")

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_send_typing_loop(context, update.message.chat_id, stop_typing))

    try:
        # Forward the message to the Agent Hub Gateway
        # Long timeout — Rothchild analysis can take 3-5 minutes
        async with httpx.AsyncClient(timeout=GATEWAY_TIMEOUT_SECONDS) as client:
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
        file_paths = data.get("file_paths") or []

        # Stop typing indicator before sending reply
        stop_typing.set()
        await typing_task

        # Send the main text response
        if reply_text:
            await update.message.reply_text(reply_text)

        # Send file attachments (result.md files) as documents
        if file_paths:
            await context.bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.UPLOAD_DOCUMENT)
            for fpath in file_paths:
                if not os.path.exists(fpath):
                    logger.warning(f"File not found for attachment: {fpath}")
                    continue
                fname = os.path.basename(fpath)
                with open(fpath, "rb") as f:
                    await context.bot.send_document(
                        chat_id=update.message.chat_id,
                        document=f,
                        filename=fname,
                        caption=f"📊 {fname}",
                    )
                logger.info(f"Sent document: {fname}")

        # Legacy: chunked text reports (kept for backward compatibility, will be empty going forward)
        reports = data.get("reports") or []
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
                if not chunk:
                    continue
                while len(chunk) > MAX_LEN:
                    await update.message.reply_text(chunk[:MAX_LEN])
                    chunk = chunk[MAX_LEN:]
                if chunk:
                    await update.message.reply_text(chunk)

    except httpx.ReadTimeout:
        stop_typing.set()
        await typing_task
        logger.error(f"Gateway timed out after {GATEWAY_TIMEOUT_SECONDS}s for chat {chat_id}")
        await update.message.reply_text(
            "⏳ The analysis is taking longer than expected. "
            "Please try: *'What's the status of my analysis?'* in a minute.",
            parse_mode="Markdown"
        )
    except Exception as e:
        stop_typing.set()
        await typing_task
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
