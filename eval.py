#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_fn(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def regex_match_fn(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(pattern,
                              flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException as e:
        print('Regular expression failed to compile: %s' % pattern)
        print(e)
        return False
    return compiled.match(prediction) is not None


def get_one_em_f1(predicted_text, ground_truths):
    em = metric_max_over_ground_truths(exact_match_fn, predicted_text,
                                       ground_truths)
    f1 = metric_max_over_ground_truths(qa_f1_score, predicted_text,
                                       ground_truths)
    return em, f1


def get_em_f1(all_predictions, all_ground_truths):
    f1_score_cum = em_score_cum = 0
    total = len(all_predictions)
    for orig_id, predicted_text in all_predictions.items():
        ground_truths = all_ground_truths[orig_id]
        em, f1 = get_one_em_f1(predicted_text, ground_truths)
        em_score_cum += em
        f1_score_cum += f1
    em_score_cum = 100.0 * em_score_cum / total
    f1_score_cum = 100.0 * f1_score_cum / total
    return em_score_cum, f1_score_cum


def get_example_and_dev_ids(feature_id):
    # e.g. 6 for ex1000_56be4db0acb8001400a502ec
    separator_idx = feature_id.find('_')
    para_sep_idx = feature_id.find(':')  # for squad compatibility
    if para_sep_idx == -1:
        para_sep_idx = separator_idx
    return int(feature_id[2:separator_idx]), feature_id[para_sep_idx + 1:]


def load_examples(input_file):
    import tensorflow as tf

    all_examples = dict()  # id to example map
    if not tf.io.gfile.exists(input_file):
        raise ValueError('{} does not exist!'.format(input_file))

    with tf.io.gfile.GFile(input_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            feature_id = item['feature_id']  # ex1_56be4db0acb8001400a502ec
            pred_id, orig_id = get_example_and_dev_ids(feature_id)
            item['orig_id'] = orig_id
            all_examples[pred_id] = item
    return all_examples
