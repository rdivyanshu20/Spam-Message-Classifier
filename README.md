# Spam Message Classifier

A beginner machine learning project that classifies short text messages as:

- `spam`
- `ham` (normal message)

## Project Goal

The goal is to teach you how a machine can learn patterns from text.

Instead of looking at numbers like `hours studied` or flower measurements, this model looks at words inside a message and learns which words are more common in spam and which words are more common in normal messages.

For example:

- messages with words like `free`, `win`, `prize`, and `claim` often look like spam
- messages with words like `class`, `call`, `meeting`, and `home` often look like normal messages

## What You Will Learn

By completing this project, you will understand:

- what a text classification problem is
- how labeled text data is stored in a dataset
- how text is split into words using tokenization
- how a `Naive Bayes` model works at a basic level
- how to split data into training and testing sets
- how to measure model accuracy
- how to test the model on new messages

## Machine Learning Concept

This project uses a simple `Multinomial Naive Bayes` style approach.

The model:

1. Counts how often each word appears in spam messages
2. Counts how often each word appears in ham messages
3. Uses probability to decide which label is more likely for a new message

It is called "naive" because it makes a simple assumption that words contribute independently, which is not perfectly true in real language, but it works surprisingly well for beginner text problems.

## Project Structure

```text
project_3_spam_classifier/
|-- README.md
|-- data/
|   `-- spam_messages.csv
`-- src/
    `-- spam_classifier.py
```

## Files Explained

### `data/spam_messages.csv`

This is the dataset.

Each row has:

- a `label`
- a `message`

Example:

```csv
spam,win a free prize now
ham,call me when you reach home
```

### `src/spam_classifier.py`

This is the main Python script.

It:

- loads the dataset
- cleans and tokenizes text
- splits the data into training and testing sets
- trains the Naive Bayes model
- predicts labels on test messages
- prints accuracy and example predictions

## How The Program Works

Here is the full workflow in simple terms:

1. Read the CSV file
2. Shuffle the dataset so the split is fair
3. Use most rows for training and a few rows for testing
4. Break each message into lowercase words
5. Count which words appear in spam and ham
6. Use those counts to score each test message
7. Pick the label with the higher score
8. Measure how many predictions were correct

## Example Output

When the script runs, it prints:

- total dataset size
- training and testing row counts
- predictions for test messages
- final test accuracy
- predictions for new example messages

Example kinds of predictions:

```text
Message: free prize waiting for you
Predicted label: spam

Message: can you call me after class
Predicted label: ham
```

## Why This Project Is Good For Beginners

- it introduces machine learning on text
- the dataset is small and easy to inspect
- the code is short enough to understand fully
- the model teaches probability in a practical way
- it helps you move toward NLP projects later

## Limitations Of This Small Project

This project is intentionally simple, so it has some limits:

- the dataset is very small
- the messages are short and clean
- the model only looks at words, not word order
- real spam detection systems use much larger datasets and better preprocessing

That is okay. The goal here is to understand the basics first.

## Good Practice Tasks

Try these one by one:

1. Add 10 to 20 more messages to the CSV file
2. Test your own custom messages
3. Print the most frequent spam words
4. Print the most frequent ham words
5. Change the train/test split ratio
6. Remove common words like `the`, `is`, and `to`
7. Rebuild the same project later with `scikit-learn`

## Questions To Think About

- Why does the word `free` strongly suggest spam?
- Why is `call me after class` probably ham?
- What happens if a message contains both spam-like and normal words?
- Why do we test on messages the model did not train on?
