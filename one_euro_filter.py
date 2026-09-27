# one_euro_filter.py
import math

class LowPassFilter:
    def __init__(self, alpha):
        self.__set_alpha(alpha)
        self.__y = None
        self.__s = None

    def __set_alpha(self, alpha):
        self.__alpha = alpha

    def filter(self, value, alpha=None):
        if alpha is not None:
            self.__set_alpha(alpha)
        if self.__s is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.__min_cutoff = float(min_cutoff)
        self.__beta = float(beta)
        self.__d_cutoff = float(d_cutoff)
        self.__x = LowPassFilter(self.__alpha(self.__min_cutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__d_cutoff))
        self.__t_prev = float(t0)
        self.__x.filter(x0)
        self.__dx.filter(dx0)

    def __alpha(self, cutoff):
        te = 1.0 / 60.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t):
        t_e = t - self.__t_prev
        if t_e <= 0.0:
            return x

        self.__t_prev = t
        dx = (x - self.__x._LowPassFilter__s) / t_e if self.__x._LowPassFilter__s is not None else 0.0
        edx = self.__dx.filter(dx, self.__alpha(self.__d_cutoff))
        cutoff = self.__min_cutoff + self.__beta * math.fabs(edx)
        return self.__x.filter(x, self.__alpha(cutoff))