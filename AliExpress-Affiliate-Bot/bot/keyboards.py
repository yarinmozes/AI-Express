"""
keyboards.py — Inline keyboard builders for Telegram messages.
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from bot.messages import CLARIFICATION_SKIP_BUTTON

# Callback data constant — matched in main.py's CallbackQueryHandler pattern.
CALLBACK_SKIP_CLARIFICATION = "skip_clarification"


def skip_clarification_keyboard() -> InlineKeyboardMarkup:
    """Build an inline keyboard with a single 'skip clarification' button.

    Returns:
        InlineKeyboardMarkup with one button that fires CALLBACK_SKIP_CLARIFICATION.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(CLARIFICATION_SKIP_BUTTON, callback_data=CALLBACK_SKIP_CLARIFICATION)]
    ])
