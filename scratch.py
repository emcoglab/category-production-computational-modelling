from typing import List


def gli() -> List[int]: return [1, 2, 3]


def gls() -> List[str]:
    return ["1", "2", "3"]


list1 = gli()
list2 = gls()
list3: List[object] = list1 + list2
