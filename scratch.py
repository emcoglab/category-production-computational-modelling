class Base:
    def __init__(self):
        self.x = set()

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value


class Derived(Base):

    @property
    def x(self):
        return set(i[0] for i in self.__x)

    @x.setter
    def x(self, value):
        self.__x = set((i, 9) for i in value)

    def print_x(self):
        print(self.__x)


if __name__ == '__main__':
    b = Base()
    b.x = {(1,0), (2,0), (3,0)}
    print(b.x)

    d = Derived()
    d.x = {1, 2, 3}
    d.print_x()
