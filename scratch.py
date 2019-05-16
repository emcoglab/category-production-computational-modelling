from dataclasses import dataclass


@dataclass
class A:
    t: int


@dataclass
class B(A):
    @classmethod
    def from_other(cls, other: A) -> 'B':
        return cls(t=other.t)


@dataclass
class C(B):
    pass


def main():
    a = A(t=1)
    b = B.from_other(a)
    c = C.from_other(a)

    print(type(a))
    print(type(b))
    print(type(c))


if __name__ == '__main__':
    main()
