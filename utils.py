# utils.py

import logging
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, List


def format_results(
    model_name: str,
    digit_stats: Dict[int, List[int]],
    total_correct: int,
    total_samples: int
) -> str:
    formatted_result = []
    formatted_result.extend([
        '',
        '_' * 30,
        f"Results for {model_name}:",
        '_' * 30,
        "Digit | Correct | Total | Accuracy",
        '_' * 30
    ])
    for digit, (correct_guesses, total_guesses) in digit_stats.items():
        accuracy = (
            (correct_guesses / total_guesses) * 100
            if total_guesses > 0
            else 0
        )
        formatted_result.append(
            f"{digit:5d} | {correct_guesses:7d} | {total_guesses:5d} | "
            f"{accuracy:7.2f}%"
        )

    total_accuracy = (total_correct / total_samples) * 100
    formatted_result.extend([
        '-' * 30,
        f"Total accuracy: {total_accuracy:.2f}%",
    ])

    return '\n\t'.join(formatted_result)


def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO
) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
