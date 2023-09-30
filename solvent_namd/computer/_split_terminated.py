from typing import NamedTuple, List


class TerminationSplit(NamedTuple):
    successful: int
    terminated: int

def split_terminated(lst: List[int]) -> TerminationSplit:
    l = len(lst)
    s = lst.count(0)
    t = l - s
    return TerminationSplit(s, t)
