"""
main.py — Entry point for the AliExpress Affiliate Bot.

Initializes the Telegram Application and registers all handlers.
"""

import logging
import os

from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from bot.handlers import search_handler, skip_callback_handler, start_command
from bot.keyboards import CALLBACK_SKIP_CLARIFICATION

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Build and start the Telegram bot with polling."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in the environment.")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(
        CallbackQueryHandler(skip_callback_handler, pattern=f"^{CALLBACK_SKIP_CLARIFICATION}$")
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_handler))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
