

__all__ = ['quadratic_interpolate', 'linear_interpolate']


def quadratic_interpolate(t1, t2, t3, x1, x2, x3, t=None, return_fit=False):
    numerator = x1 * (t2 - t3) + x2 * (t3 - t1) + x3 * (t1 - t2)
    denominator = (t1 - t2) * (t1 - t3) * (t2 - t3)
    a = numerator / denominator

    numerator = x1 * (t2 ** 2 - t3 ** 2) + x2 * (t3 ** 2 - t1 ** 2) + x3 * (t1 ** 2 - t2 ** 2)
    b = -numerator / denominator

    numerator = x1 * t2 * t3 * (t2 - t3) + x2 * t3 * t1 * (t3 - t1) + x3 * t1 * t2 * (t1 - t2)
    c = numerator / denominator

    return_t = t is None
    if t is None:
        t = -b / (2 * a)

    x = a * t ** 2 + b * t + c
    if return_t:
        if return_fit:
            return x, t, a, b, c
        return x, t
    if return_fit:
        return x, a, b, c
    return x


def linear_interpolate(t1, t2, x1, x2, t=None):
    a = (x2 - x1) / (t2 - t1)
    b = x1 - a * t1
    if t is None:
        return -b / a
    return a * t + b
