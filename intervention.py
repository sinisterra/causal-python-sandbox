import random
import pandas as pd

random.seed(42)

N = 5000


def model1(do_x=None, do_y=None, do_z=None):
    x = random.random() if do_x is None else do_x
    y = (x + 1 + (3 ** (0.5)) * random.random()) if do_y is None else do_y
    return (x, y)


def model2(do_x=None, do_y=None, do_z=None):
    y = (1 + 2 * random.random()) if do_y is None else do_y
    x = ((y - 1) / 4) + ((3 ** 0.5) * random.random() / 2) if do_x is None else do_x
    return (x, y)


# x <- z -> y
def model3(do_x=None, do_y=None, do_z=None):
    z = random.random() if do_z is None else do_z
    y = z + 1 + ((3 ** 0.5) * random.random()) if do_y is None else do_y
    x = z if do_x is None else do_x
    return (x, y, z)


def make_m1(do_x=None):
    x_ = []
    y_ = []
    for _ in range(N):
        (x, y) = model1(do_x=do_x)
        x_.append(x)
        y_.append(y)

    return pd.DataFrame({"x": x_, "y": y_})


def make_m2(do_x=None):
    x_ = []
    y_ = []
    for _ in range(N):
        (x, y) = model2(do_x=do_x)
        x_.append(x)
        y_.append(y)
    return pd.DataFrame({"x": x_, "y": y_})


def make_m3(do_x=None, do_y=None, do_z=None):
    x_ = []
    y_ = []
    z_ = []
    for _ in range(N):
        (x, y, z) = model3(do_x=do_x, do_y=do_y, do_z=do_z)
        x_.append(x)
        y_.append(y)
        z_.append(z)
    return pd.DataFrame({"x": x_, "y": y_, "z": z_})


# print(make_m1(do_x=3))
# print(make_m2(do_x=3))
make_m2(do_x=3).to_csv("./datasets/intervention_xyz.csv", index=False)
