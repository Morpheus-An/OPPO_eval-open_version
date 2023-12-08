from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Tuple, TextIO, Callable, Literal
import random
import math
import copy

import json
import numpy as np
import re

from utils import LLM
from utils import Metric
from utils import Functions

from tqdm import tqdm
_ScoreType = Union[bool, float, int]

BOLD_GREEN = "\033[1;32m"
BOLD_RED = "\033[1;31m"
RESET_COLOR = "\033[0m"

ScoreType = Union[_ScoreType, List[_ScoreType],
                  List[List[_ScoreType]], Dict[str, 'ScoreType']]
EvalFuncType = Callable[[List[List[Any]]], Union[ScoreType, str]]

EvalFuncDict = {}


def EvalFunc(name: Optional[str] = None):
    def wrapper(func: EvalFuncType) -> EvalFuncType:
        nonlocal name
        if name is None or name == '':
            name = func.__name__
        EvalFuncDict[name] = func
        return func
    return wrapper


def clean_newlines(text):
    """clean all empty lines and strip the text"""
    return re.sub(r'\n+', '\n', text).strip()


def time_now():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_time


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def clear_punctuations(s):
    '''
    clear all character that is not a English letter(\\w), space(\\s) or Chinese character.
    '''
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', s)
