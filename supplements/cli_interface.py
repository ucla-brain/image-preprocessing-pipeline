from typing import Tuple, Callable, Union, List
from math import inf
from os import getenv
from datetime import datetime


def is_pycharm():
    return getenv("PYCHARM_HOSTED") is not None


def select_among_multiple_options(
        question: str,
        options: list,
        return_index=False
):
    if len(options) > 1:
        choices = list(map(str, range((0 if return_index else 1), len(options) + (0 if return_index else 1))))
        choices_print = f"\n\n{question}"
        for idx, option in enumerate(options, start=0 if return_index else 1):
            choices_print += f"\n\t{idx} = {option}"
        choices_print = choices_print + "\n\n"
        choice = str()
        while choice not in choices:
            choice = input(choices_print).strip()
        return choice if return_index else options[int(choice) - 1]
    elif len(options) == 1:
        return "0" if return_index else options[0]
    else:
        print("no options provided")
        raise RuntimeError


def ask_true_false_question(prompt_message: str) -> bool:
    answer = ''
    while answer not in {"1", "2"}:
        answer = input(
            f'\n\n{prompt_message}\n'
            '\t1 = Yes\n'
            '\t2 = No\n').strip()
    return answer == "1"


def ask_for_a_number_in_range(message: str, range_val: Union[Tuple[int, int], Tuple[float, float]], dtype: Callable):
    answer = inf
    while not range_val[0] <= answer <= range_val[1]:
        try:
            answer = dtype(input(f'\n\n{message}\n').strip())
        except ValueError:
            answer = inf
    return answer


def select_multiple_among_list(question: str, input_list: List[str]) -> List[str]:
    output_list = []
    if len(input_list) == 1:
        output_list = input_list.copy()
    else:
        for item in input_list:
            if ask_true_false_question(f"Do you need to apply {question} to {item} channel?"):
                output_list += [item]
    return output_list


class PrintColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def date_time_now():
    return datetime.now().isoformat(timespec='seconds', sep=' ')
