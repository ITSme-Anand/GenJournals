import re

# Define the abbreviations dictionary
abbreviations = {
    "idk": "I don't know",
    "brb": "be right back",
    "lol": "laugh out loud",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "btw": "by the way",
    "imo": "in my opinion",
    "fyi": "for your information",
    "asap": "as soon as possible",
    "smh": "shaking my head",
    "afk": "away from keyboard",
    "bff": "best friend forever",
    "tbh": "to be honest",
    "np": "no problem",
    "dm": "direct message",
    "ikr": "I know right"
}

# Define a preprocessing function to replace abbreviations in the user's prompt
def preprocess_text(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b', re.IGNORECASE)
    processed_text = pattern.sub(lambda match: abbreviations[match.group(0).lower()], text)
    return processed_text

# Function to adjust the response tone
def handle_sad_message(response_text):
    if response_text.lower().startswith(("i'm sorry", "i am sorry", "i apologize", "sorry to hear")):
        motivational_start = "You're stronger than you realize. It’s okay to feel down sometimes, but I believe you can get through this."
        response_text = re.sub(r"^(i'm sorry|i am sorry|sorry to hear|i apologize).*", motivational_start, response_text, flags=re.IGNORECASE)
    return response_text
