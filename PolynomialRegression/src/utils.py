


def power_sum(X, n):
    if n == 0 : return 1
    return X**n + power_sum(X, n -1)