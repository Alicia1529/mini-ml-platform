"""Show how five messages become three batches of sizes 2, 2, and 1."""

from mini_ml_platform.batch_inference import split_batches
from mini_ml_platform.moderation import Message


if __name__ == "__main__":
    texts = ["hello", "SCAM", "nice", "send PASSWORD", "hackathon"]
    messages = [Message(i, text) for i, text in enumerate(texts)]
    batches = split_batches(messages, 2)
    print("Batches of messages:", batches)
    print([[message.text for message in batch] for batch in batches])
