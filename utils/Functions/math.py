from typing import List, Any


def average(x_list: List[Any], basic):
    if x_list:
        return sum(x_list) / len(x_list)
    else:
        return basic

