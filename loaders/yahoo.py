import random

TOPIC_NAMES = {
    0: "Society_Culture",
    1: "Science_Mathematics",
    2: "Health",
    3: "Education_Reference",
    4: "Computers_Internet",
    5: "Sports",
    6: "Business_Finance",
    7: "Entertainment_Music",
    8: "Family_Relationships",
    9: "Politics_Government",
}


def sample_yahoo_by_topic(n: int, seed: int = 42, split: str = "train") -> dict[str, list[str]]:
    """
    Sample n texts per topic from community-datasets/yahoo_answers_topics.

    Returns a dict {topic_name: [text, ...]} for all 10 topics.
    Each text is question_title + " " + question_content.
    """
    from datasets import load_dataset

    ds = load_dataset("community-datasets/yahoo_answers_topics", split=split, trust_remote_code=True)

    # Group text by topic id
    buckets: dict[int, list[str]] = {i: [] for i in TOPIC_NAMES}
    for row in ds:
        tid = int(row["topic"])
        title = str(row.get("question_title") or "")
        body  = str(row.get("question_content") or "")
        text  = (title + " " + body).strip()
        if text:
            buckets[tid].append(text)

    rng = random.Random(seed)
    return {
        TOPIC_NAMES[tid]: rng.sample(texts, min(n, len(texts)))
        for tid, texts in buckets.items()
    }
