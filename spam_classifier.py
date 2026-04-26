from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter
from pathlib import Path


MessageRow = tuple[str, str]


def load_dataset(file_path: Path) -> list[MessageRow]:
    """Load message labels and text from a CSV file."""
    rows: list[MessageRow] = []

    with file_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append((row["label"], row["message"]))

    return rows


def split_dataset(rows: list[MessageRow], train_ratio: float = 0.75) -> tuple[list[MessageRow], list[MessageRow]]:
    """Shuffle the dataset in a repeatable way, then split it into training and testing sets."""
    shuffled_rows = rows[:]
    random.Random(42).shuffle(shuffled_rows)

    split_index = int(len(shuffled_rows) * train_ratio)
    training_rows = shuffled_rows[:split_index]
    testing_rows = shuffled_rows[split_index:]
    return training_rows, testing_rows


def tokenize(text: str) -> list[str]:
    """Lowercase the text and keep only simple word tokens."""
    return re.findall(r"[a-z]+", text.lower())


def train_naive_bayes(rows: list[MessageRow]) -> dict[str, object]:
    """
    Train a simple multinomial Naive Bayes model.

    The model learns:
    - prior probability of each class
    - word counts for spam and ham
    - total number of words in each class
    - full vocabulary
    """
    class_counts: Counter[str] = Counter()
    word_counts_by_class: dict[str, Counter[str]] = {
        "spam": Counter(),
        "ham": Counter(),
    }
    total_words_by_class: dict[str, int] = {
        "spam": 0,
        "ham": 0,
    }
    vocabulary: set[str] = set()

    for label, message in rows:
        class_counts[label] += 1
        tokens = tokenize(message)

        for token in tokens:
            word_counts_by_class[label][token] += 1
            total_words_by_class[label] += 1
            vocabulary.add(token)

    return {
        "class_counts": class_counts,
        "word_counts_by_class": word_counts_by_class,
        "total_words_by_class": total_words_by_class,
        "vocabulary": vocabulary,
        "total_rows": len(rows),
    }


def score_message(message: str, label: str, model: dict[str, object]) -> float:
    """Calculate the log-probability score of a message for one class."""
    class_counts = model["class_counts"]
    word_counts_by_class = model["word_counts_by_class"]
    total_words_by_class = model["total_words_by_class"]
    vocabulary = model["vocabulary"]
    total_rows = model["total_rows"]

    tokens = tokenize(message)
    vocabulary_size = len(vocabulary)

    prior = class_counts[label] / total_rows
    score = math.log(prior)

    for token in tokens:
        token_count = word_counts_by_class[label][token]
        token_probability = (token_count + 1) / (total_words_by_class[label] + vocabulary_size)
        score += math.log(token_probability)

    return score


def predict_label(message: str, model: dict[str, object]) -> str:
    """Predict whether a message is spam or ham."""
    spam_score = score_message(message, "spam", model)
    ham_score = score_message(message, "ham", model)
    return "spam" if spam_score > ham_score else "ham"


def accuracy_score(actual_labels: list[str], predicted_labels: list[str]) -> float:
    """Calculate classification accuracy."""
    correct_predictions = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted)
    return correct_predictions / len(actual_labels)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "spam_messages.csv"

    rows = load_dataset(dataset_path)
    training_rows, testing_rows = split_dataset(rows)
    model = train_naive_bayes(training_rows)

    actual_labels: list[str] = []
    predicted_labels: list[str] = []

    print("Spam Message Classifier")
    print("-" * 30)
    print(f"Total rows: {len(rows)}")
    print(f"Training rows: {len(training_rows)}")
    print(f"Testing rows: {len(testing_rows)}")
    print("Model: Naive Bayes")
    print()
    print("Test predictions:")

    for actual_label, message in testing_rows:
        predicted_label = predict_label(message, model)
        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)

        print(
            f"Actual: {actual_label:<4} | "
            f"Predicted: {predicted_label:<4} | "
            f"Message: {message}"
        )

    accuracy = accuracy_score(actual_labels, predicted_labels)
    print()
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print()

    example_messages = [
        "free prize waiting for you",
        "can you call me after class",
        "urgent claim your bonus now",
    ]

    print("New message examples:")
    for message in example_messages:
        predicted_label = predict_label(message, model)
        print(f"Message: {message}")
        print(f"Predicted label: {predicted_label}")
        print()


if __name__ == "__main__":
    main()
