"""Evaluation tools"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_pie(xpos, ypos, dist, colors=None, ax=None, **kwargs):
    """Draw a pie chart at position `xpos`, `ypos` with distribution `dist`.

    Args:
        xpos (float): x position
        ypos (float): y position
        dist (list): list of values to be pie charted
        colors (list): list of colors for each pie slice
        ax (matplotlib.axes.Axes): axes to draw on
        **kwargs: keyword arguments to pass to `ax.scatter`

    Returns:
        matplotlib.axes.Axes: axes with pie chart drawn

    """

    if ax is None:
        fig, ax = plt.subplots()

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()

    if colors is None:
        colors = itertools.repeat(None, len(dist))
    for r1, r2, facecolor in zip(pie[:-1], pie[1:], colors):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])

        ax.scatter([xpos], [ypos], marker=xy, facecolor=facecolor, **kwargs)

    return ax


def recurse_mean_dict(d, d_mean):
    """Recursively mean the values of dict `d` of lists and append into dict
    `d_mean` of lists.

    `d_mean` must be initialized with the same structure as `d`.

    Args:
        d (dict): nested dictionary of lists
        d_mean (dict): nested dictionary of lists, initialized the same as d.

    """
    for k, v in d.items():
        if isinstance(v, dict):
            recurse_mean_dict(v, d_mean[k])
        else:
            d_mean[k].append(np.mean(v))


def recurse_dict_to_numpy(d):
    for k, v in d.items():
        if isinstance(v, dict):
            recurse_dict_to_numpy(d[k])
        elif torch.is_tensor(v):
            d[k] = v.detach().cpu().numpy()


def recurse_avg_dict(d, d_avg, div_by_weights=False):
    """Recursively average the values of dict `d` of lists and append into dict
    `d_avg` of lists.

    `d_avg` must be initialized with the same structure as `d`. Base-level dicts
    must contain a list with key "weights".

    Args:
        d (dict): nested dict of lists
        d_avg (dict): nested dict of lists, initialized the same as d.

    """
    for k, v in d.items():
        if isinstance(v, dict):
            recurse_avg_dict(v, d_avg[k], div_by_weights)
        else:
            if div_by_weights and k not in {"lr", "weights"}:
                d_avg[k].append(np.average(np.divide(v, d["weights"]), weights=d["weights"]))
            else:
                d_avg[k].append(np.average(v, weights=d["weights"]))


def recurse_running_dict(d, d_hist):
    """Recursively append the values of a dict `d` into dict `d_hist` of lists.

    `d_hist` will contain lists. The output dict must be initialized with the
    same structure as `d`, but with lists instead of values.

    Args:
        d (dict): nested dict of lists
        d_hist (dict): nested dict of lists, initialized the same as d but with
            lists.

    """
    for k, v in d.items():
        if isinstance(v, dict):
            recurse_running_dict(v, d_hist[k])
        else:
            d_hist[k].append(v)


def format_iters(nested_list, startpoint=False, endpoint=True):
    """Generates x and y values, given a nested list of iterations by epoch.


    x will be evenly spaced by epoch, and y will be the flattened values in the
    nested list.

    Args:
        nested_list (list): List of lists.
        startpoint (bool): Include startpoint of iteration. Defaults to False.
        endpoint (bool): Include endpoint of iteration. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y values.


    """

    x = []
    if startpoint:
        for i, l in enumerate(nested_list):
            if endpoint and i == len(nested_list) - 1:
                x_i = np.linspace(i - 1, i, len(l), endpoint=True, dtype=np.float32)
            else:
                x_i = np.linspace(i - 1, i, len(l), endpoint=False, dtype=np.float32)
            x.append(x_i)
    else:
        for i, l in enumerate(nested_list):
            if not endpoint and i == len(nested_list) - 1:
                x_i = np.linspace(i, i - 1, len(l + 1), endpoint=False, dtype=np.float32)
                x_i = x_i[1:]
            else:
                x_i = np.linspace(i, i - 1, len(l), endpoint=False, dtype=np.float32)

            # Flip to not include startpoint i.e. shift to end of iteration
            x_i = np.flip(x_i)
            x.append(x_i)

    x = np.asarray(list(itertools.chain(*x)))
    y = np.asarray(list(itertools.chain(*nested_list)))

    return x, y
