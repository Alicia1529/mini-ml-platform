def is_unsafe(text: str) -> bool:
    text = text.lower()
    unsafe_words = {'scam', 'password', 'hack'}
    for word in unsafe_words:
        if word in text:
            return True
    return False
