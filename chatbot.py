"""Interactive chatbot runner.

Improved robustness, logging, and argument support compared to the original script.
Loads training artifacts from the repository root by default. If a full saved model
(`chatbot_model.h5`) exists it will be used; otherwise the code will attempt to
reconstruct the model from saved pickles and weights.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

DEFAULT_INTENTS = os.path.join(os.path.dirname(__file__), 'intents.json')
DEFAULT_WORDS = os.path.join(os.path.dirname(__file__), 'words.pkl')
DEFAULT_CLASSES = os.path.join(os.path.dirname(__file__), 'classes.pkl')
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), 'chatbot_model.h5')
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'chatbotmodel_weights.h5')


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def ensure_nltk() -> None:
    # Download required NLTK data if not present. Some environments need 'punkt_tab'
    # or the open multilingual wordnet ('omw-1.4'). Try to download these if missing
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except Exception:
            pass
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        try:
            nltk.download('omw-1.4')
        except Exception:
            pass


def load_artifacts(words_path: str, classes_path: str, intents_path: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents = json.load(f)

    with open(words_path, 'rb') as f:
        words = pickle.load(f)

    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)

    return words, classes, intents


def normalize_text(text: str) -> str:
    # basic normalization: lowercase and remove extra non-word characters
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence: str) -> List[str]:
    sentence = normalize_text(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in sentence_words]


def bag_of_words(sentence: str, words: List[str]) -> np.ndarray:
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag, dtype=int)


def build_dummy_model(input_size: int, output_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(input_size,), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
    return model


def load_model_safe(model_path: str, weights_path: str, words: List[str], classes: List[str]) -> tf.keras.Model:
    # Prefer a single-file saved model if available
    if os.path.exists(model_path):
        logging.info('Loading full model from %s', model_path)
        return tf.keras.models.load_model(model_path)

    logging.info('Full model not found, attempting to load architecture and weights...')
    model = build_dummy_model(len(words), len(classes))
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        logging.info('Weights loaded from %s', weights_path)
    else:
        logging.warning('Weights file not found at %s. The model will be untrained.', weights_path)
    return model


def predict_class(sentence: str, model: tf.keras.Model, words: List[str], classes: List[str], threshold: float = 0.25):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': float(r[1])} for r in results]


def get_response(intents_list: List[Dict[str, Any]], intents_json: Dict[str, Any]) -> str:
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"
    tag = intents_list[0]['intent']
    for i in intents_json.get('intents', []):
        if i.get('tag') == tag:
            return random.choice(i.get('responses', ['Sorry, I have no response configured.']))
    return "Sorry, something went wrong finding a response."


def interactive_loop(model: tf.keras.Model, words: List[str], classes: List[str], intents_json: Dict[str, Any]) -> None:
    print('Go! Bot is running! (type "quit" or "exit" to stop)')
    while True:
        try:
            message = input('You: ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\nGoodbye!')
            break
        if not message:
            continue
        if message.lower() in ('quit', 'exit'):
            print('Goodbye!')
            break
        ints = predict_class(message, model, words, classes)
        res = get_response(ints, intents_json)
        print('Bot:', res)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the interactive chatbot')
    parser.add_argument('--intents', '-i', default=DEFAULT_INTENTS, help='Path to intents.json')
    parser.add_argument('--words', '-w', default=DEFAULT_WORDS, help='Path to words.pkl')
    parser.add_argument('--classes', '-c', default=DEFAULT_CLASSES, help='Path to classes.pkl')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL, help='Path to saved model (.h5)')
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS, help='Path to weights file (.h5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    setup_logging(args.verbose)
    ensure_nltk()

    if not (os.path.exists(args.words) and os.path.exists(args.classes) and os.path.exists(args.intents)):
        logging.error('Required artifact missing. Make sure training has been run and words/classes pickles and intents.json exist.')
        raise SystemExit(1)

    words, classes, intents_json = load_artifacts(args.words, args.classes, args.intents)
    model = load_model_safe(args.model, args.weights, words, classes)
    interactive_loop(model, words, classes, intents_json)


if __name__ == '__main__':
    main()
