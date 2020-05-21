from typing import FrozenSet


class Base:
    def __init__(self, items=None):
        self.items: FrozenSet = frozenset() if items is None else items

    @property
    def items(self) -> FrozenSet:
        print("Base getter")
        return self.__items

    @items.setter
    def items(self, value):
        print("Base setter")
        self.__items = value


class Derived(Base):

    @property
    def items(self) -> FrozenSet:
        print("Derived getter")
        return self.__items

    @items.setter
    def items(self, value):
        print("Derived setter")
        self.__items = value


if __name__ == '__main__':
    d = Derived(frozenset({0, 1, 2}))
    d.items -= {0}
    print(d.items)
