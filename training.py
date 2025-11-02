"""Training script for the simple intents-based chatbot.

Saves
- `words.pkl` and `classes.pkl` (vocab and labels)
- `chatbot_model.h5` (full Keras model)
- `chatbotmodel_weights.h5` (weights backup)
- `training_history.pkl` (history dict)

Usage: python training.py --epochs 200 --batch-size 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from typing import List

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf


def ensure_nltk() -> None:
    # Ensure core tokenizers and corpora are available. Some NLTK versions expect
    # the "punkt_tab" subresource; try to download both to be robust.
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    # punkt_tab is referenced by some punkt implementations — download if missing
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except Exception:
            # not all NLTK distributions expose punkt_tab via downloader; ignore
            pass
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    # Optional wordnet data used by newer NLTK versions
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        try:
            nltk.download('omw-1.4')
        except Exception:
            pass


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def build_model(input_size: int, output_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(input_size,), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
    return model


def train(intents_path: str, epochs: int = 200, batch_size: int = 5, verbose: int = 1) -> None:
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents = json.load(f)

    lemmatizer = WordNetLemmatizer()

    words: List[str] = []
    classes: List[str] = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents.get('intents', []):
        for pattern in intent.get('patterns', []):
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent.get('tag')))
            if intent.get('tag') not in classes:
                classes.append(intent.get('tag'))

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    # persist vocabulary and classes
    with open('words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for w in words:
            bag.append(1 if w in word_patterns else 0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    train_x = np.array([x[0] for x in training])
    train_y = np.array([x[1] for x in training])

    model = build_model(len(train_x[0]), len(train_y[0]))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    hist = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Save artifacts
    model.save('chatbot_model.h5')
    model.save_weights('chatbotmodel_weights.h5')
    with open('training_history.pkl', 'wb') as history_file:
        pickle.dump(hist.history, history_file)

    logging.info('Saved model to chatbot_model.h5 and weights to chatbotmodel_weights.h5')


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the intents-based chatbot')
    parser.add_argument('--intents', '-i', default='intents.json', help='Path to intents.json')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=5, help='Training batch size')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging output')
    args = parser.parse_args()

    setup_logging(args.verbose)
    ensure_nltk()

    if not os.path.exists(args.intents):
        logging.error('intents.json not found at %s', args.intents)
        raise SystemExit(1)

    train(args.intents, epochs=args.epochs, batch_size=args.batch_size, verbose=1)


if __name__ == '__main__':
    main()
