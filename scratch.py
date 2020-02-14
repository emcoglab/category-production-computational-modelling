class Base:
    def __init__(self):
        self.x = set()


class Derived(Base):

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value
        self.__y = len(self.__x) + 10_000

    @property
    def y(self):
        return self.__y


if __name__ == '__main__':
    b = Base()
    b.x = set(range(10))

    d = Derived()
    print(d.y)
    d.x = set(range(10))
    print(d.y)
