import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)

cmap_data = plt.cm.tab20
cmap_classes = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4

# This module heavily uses functions from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py

def calculate_groups(n_rows, n_groups):
    """Calculate n groups out of n rows

    :param n_rows: number of rows in the data
    :type n_rows: int
    :param n_groups: number of groups in the data
    :type n_groups: int
    :return: Array where each element has a group assignment (int) value and its index corresponds to an index out of n_rows
    :rtype: numpy.array
    """
    return np.hstack([[ii] * int(n_rows/n_groups) for ii in range(n_groups)])


def visualize_groups(classes, groups, name, lw=100):
    """Visulize class distribution in the data, and how an evenly split grouping might look

    :param classes: vector of the class labels
    :type classes: numpy.array
    :param groups: vector of group assignments for each label
    :type groups: numpy.array
    :param name: Title of plot
    :type name: str
    :param lw: Width of the line being plotted, defaults to 100
    :type lw: int, optional
    """
    # Visualize dataset groups
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
               lw=lw, cmap=cmap_data)
    ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
               lw=lw, cmap=cmap_classes)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Data index")
    ax.set_title(name, fontsize=15)


def plot_cv_indices(cv, X, y, group, ax, n_splits=10, lw=10):
    """Create a sample plot for indices of a cross-validation object.

    :param cv: Cross validation object
    :type cv: sklearn.model_selection
    :param X: X data 
    :type X: numpy.array
    :param y: Class labels for X
    :type y: numpy.array
    :param group: vector of group assignments for each corresponding index in X and y
    :type group: numpy.array
    :param ax: pyplot axes
    :type ax: matplotlib.axes.Axes
    :param n_splits: number of splits as passed to cv, defaults to 10
    :type n_splits: int, optional
    :param lw: Width of each line being plotted, defaults to 10
    :type lw: int, optional
    :return: Modified axes
    :rtype: matplotlib.axes.Axes
    """

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_classes)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, X.shape[0]])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax
