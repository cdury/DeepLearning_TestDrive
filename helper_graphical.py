import numpy as np
import matplotlib.pyplot as plt


#   plot the data on a figure
def plot_data(pl, data, label):
    """Plots the data according to the labels

    :param pl:      plot
    :param data:    2D data
    :param label:   Labels of data
    :return:
    """
    # plot class where y==0
    pl.plot(data[label == 0, 0], data[label == 0, 1], "ob", alpha=0.5)
    # plot class where y==1
    pl.plot(data[label == 1, 0], data[label == 1, 1], "xr", alpha=0.5)
    pl.legend(["0", "1"])
    return pl


#   Common function that draws the decision boundaries
def plot_decision_boundary(model, data, label) -> plt:
    """Plots a grid of model prediction in addition to the
        data according to the labels

    :param model:   Trained keras model
    :param data:    2D data
    :param label:   Labels of data
    :return:
    """
    amin, bmin = data.min(axis=0) - 0.1
    amax, bmax = data.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make prediction with the model and reshape the output so contourf can plot it
    c = model.predict(ab)
    z_data = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    # plot the contour
    plt.contourf(aa, bb, z_data, cmap="bwr", alpha=0.2)
    # plot the moons of data
    plot_data(plt, data, label)

    return plt
