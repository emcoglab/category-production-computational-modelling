class Base:
    def __init__(self, a):
        self.a = a
        print(f"Base.__init__({self.a})")


class Mix:
    def __init__(self, *args):
        self.mixed = 1
        print(f"Mix.__init__()")
        super().__init__(*args)


class Derived(Mix, Base):
    def __init__(self):
        print(f"Derived.__init__()")
        super().__init__(2)


if __name__ == '__main__':
    d = Derived()
    print(d.a,  d.mixed)
