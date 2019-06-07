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
