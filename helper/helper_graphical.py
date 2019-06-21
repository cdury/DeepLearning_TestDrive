import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def category_evaluation(   n_classes,LABELS, given, predictions ):
    print("Precision: {}%".format(100 * metrics.precision_score(given, predictions, average="weighted")))
    print("Recall: {}%".format(100 * metrics.recall_score(given, predictions, average="weighted")))
    print("f1_score: {}%".format(100 * metrics.f1_score(given, predictions, average="weighted")))
    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(given, predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
    print("Note: training and testing data is not equally distributed amongst classes, ")
    # Plot Results:
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

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

def plot_roc(pred,y):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    return plt

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return plt

import pandas as pd
def plot_regression_chart(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred.flatten(), 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    return plt

