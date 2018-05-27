import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Visualize:
    """Class that implement different visualization tools and ready methods for matplotlib."""
    
    def __init__(self):
        """Initialize the size of the plots."""
        plt.rcParams['figure.figsize'] = [15, 8]
    
    def plot_confusion_matrix(self, y_test, y_pred, classes, normalize=False,
                              title='Confusion Matrix', cmap=plt.cm.get_cmap('Reds')):
        """Plot a confusion matrix from the given parameters.
        
        Args:
            y_test (1darray): 1D array represeting the groundtruth values.
            y_pred (1darray): 1D array representing the predicted values.
            classes (List[str]): List of classes name.
            normalize (bool): Weather to normalize the confusion matrix or not.
            title (str): the title of the confusion matrix plot.
            cmp (matplotlib.cmap): The color map used on the confusion matrix plot.
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, round (cm[i, j],2), horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')