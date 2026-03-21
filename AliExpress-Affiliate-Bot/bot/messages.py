"""
messages.py — Hebrew UI string constants.

All user-facing text lives here. No Hebrew strings in other modules.
Uses Telegram HTML parse mode for formatting tags.
"""

WELCOME = (
    "שלום! אני AI-Express, סוכן הקניות החכם שלך. 🛍️\n"
    "איזה מוצר תרצה שאחפש עבורך?"
)

# Progress messages — edited in-place to show pipeline stages.
SEARCHING = "🔍 מחפש עבורך: {query}..."
SEARCHING_TRANSLATING = "🌐 מתרגם את השאילתה..."
SEARCHING_ALIEXPRESS = "📦 מחפש מוצרים ב-AliExpress..."
SEARCHING_AI = "🤖 AI מנתח את התוצאות עבורך..."

# Results header — {query} is the original Hebrew query.
RESULTS_HEADER = '🔍 תוצאות עבור: "<b>{query}</b>"\n\n'

# Single product card — all placeholders required.
RESULT_CARD = (
    "{label}\n"
    "<b>{title}</b>\n"
    "💵 ${price}  |  ⭐ {rating}\n"
    "<i>{reason}</i>\n"
    '🛒 <a href="{link}">לרכישה ב-AliExpress</a>'
)

RESULT_CARD_NO_RATING = (
    "{label}\n"
    "<b>{title}</b>\n"
    "💵 ${price}\n"
    "<i>{reason}</i>\n"
    '🛒 <a href="{link}">לרכישה ב-AliExpress</a>'
)

# Divider between product cards.
CARD_DIVIDER = "\n\n─────────────────\n\n"

# --- Clarification loop ---

# First question in the loop — includes the intro sentence.
CLARIFICATION_INTRO = (
    "🤔 כדי למצוא לך את המוצר הכי מתאים, יש לי כמה שאלות קצרות!\n\n"
    "שאלה {index}/{total}:\n<b>{question}</b>"
)

# Follow-up questions (2nd, 3rd, 4th).
CLARIFICATION_FOLLOWUP = "שאלה {index}/{total}:\n<b>{question}</b>"

# Skip button label (used in keyboards.py).
CLARIFICATION_SKIP_BUTTON = "דלג, פשוט תמצא לי 🔍"

# Progress message shown while merging answers into a refined search term.
CLARIFICATION_REFINING = "✨ מעבד את התשובות ומחפש את ההתאמה הכי טובה..."

# Shown when user hits Skip — replaces the question message before running the pipeline.
CLARIFICATION_SKIPPED = "🔍 מחפש עם השאילתה המקורית..."

# Appended to a product card when the LLM could not verify all hard technical specs
# from the product title alone. The text is hardcoded here — never LLM-generated.
SPEC_WARNING = (
    "⚠️ <i>לא ניתן היה לאמת את כל המפרטים הטכניים מכותרת המוצר — "
    "אנא בדוק את דף המוצר לפני הרכישה.</i>"
)

# --- Errors ---

ERROR_NO_RESULTS = "😕 לא נמצאו תוצאות עבור החיפוש שלך. נסה מילות חיפוש אחרות."
ERROR_API_FAILURE = "⚠️ אירעה שגיאה בחיפוש. אנא נסה שוב בעוד מספר שניות."
ERROR_LLM_FAILURE = "🤖 הבינה המלאכותית לא הצליחה לנתח את התוצאות. אנא נסה שוב."
