import numpy as np


def simulate_visual(x, y, p=0.2):
    """ simulate visual detection results, where some ground truth is disappeared
    :param x:           array, N x 4
    :param y:           array, N
    :param p:           scalar, probability for throw out detection
    :return:
        new_x
        new_y
    """
    x = x.copy()
    y = y.copy()
    prob = np.random.rand(len(x))
    mask = np.where(prob<= 1-p)[0]
    if len(mask) < 2:
        mask = np.random.choice(np.arange(len(x)), 2)

    return x[mask], y[mask]


def simulate_points(x, y, p=0.):
    """ simulate cloud points cluster results, where some clusters are false alarms
    :param x:           array, N x 4
    :param y:           array, N
    :param p:           the probability to create new targets
    :return:
        new_x
        new_y
    """
    x, y = simulate_visual(x, y, p=0.01)
    anchor_num = 20  # len(x)
    prob = np.random.rand(anchor_num)
    new_points = np.random.rand(anchor_num, 4)*2 - 1
    new_points[:, 2:] *= np.random.rand(anchor_num, 2)*0.3
    new_points = new_points[prob<p]
    new_x = np.concatenate((x, new_points), axis=0)
    new_y = np.concatenate((y, -1*np.arange(1, len(new_points)+1)), axis=0)
    return new_x, new_y


def RandomTranslate(x, y):
    """add some random translate to each point
    :param x:           array, N x 4
    :param y:           array, N
    :returns:
        new_x
        new_y
    """
    x[:, :2] += x[:, 2:] * (np.random.rand(len(x), 2)*2 - 1) * 0.4
    x[:, 2:] *= (1 +(np.random.rand(len(x), 2)*2 - 1) * 0.1)
    return x, y

