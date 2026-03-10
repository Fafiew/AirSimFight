"""Tiny numpy fallback for offline test environments."""
import math
import random as _random


class Array(list):
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Array([a + other for a in self])
        return Array([a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Array([a - other for a in self])
        return Array([a - b for a, b in zip(self, other)])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Array([a * other for a in self])
        return Array([a * b for a, b in zip(self, other)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Array([a / other for a in self])
        return Array([a / b for a, b in zip(self, other)])

    def copy(self):
        return Array(self[:])

    def tolist(self):
        return list(self)


def array(x, dtype=float):
    return Array([dtype(v) for v in x])


def zeros(n, dtype=float):
    return Array([dtype(0) for _ in range(n)])


def clip(x, low, high):
    if isinstance(x, (list, Array)):
        return Array([min(max(v, low), high) for v in x])
    return min(max(x, low), high)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def asarray(x, dtype=float):
    return array(x, dtype=dtype)


def concatenate(parts):
    out = []
    for p in parts:
        out.extend(list(p))
    return Array(out)


def deg2rad(x):
    return x * math.pi / 180.0


def arccos(x):
    return math.acos(x)


class _Linalg:
    @staticmethod
    def norm(v):
        return math.sqrt(sum(float(a) * float(a) for a in v))


linalg = _Linalg()


class _Random:
    @staticmethod
    def normal(mu, sigma):
        return _random.gauss(mu, sigma)


random = _Random()


pi = math.pi
inf = float("inf")
float32 = float


def cos(x):
    import math
    return math.cos(x)

def sin(x):
    import math
    return math.sin(x)


ndarray = Array
