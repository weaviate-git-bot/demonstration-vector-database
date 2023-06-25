from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

#
#       Definitions
#


class ConfusiMatrix:
    def __init__(self):
        pass

    def show(self):
        # Create confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Show confusion matrix
        ax = plt.subplot()
        sns.heatmap(conf_matrix, annot=True, ax = ax, cmap='Blues', fmt='g', cbar=False)

        # Add labels, title and ticks
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Acctual')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Benign', 'Attack'])
        ax.yaxis.set_ticklabels(['Benign', 'Attack'])