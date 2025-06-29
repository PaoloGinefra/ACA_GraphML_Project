import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def plotTrueVsPred(y, y_pred, y_val, y_val_pred, title='True vs Predicted Values', filename=None, hardLimit=True):
    """
    Plots true vs predicted values for both training and validation datasets.

    Parameters:
        y (array-like): True target values for the training set.
        y_pred (array-like): Predicted values for the training set.
        y_val (array-like): True target values for the validation set.
        y_val_pred (array-like): Predicted values for the validation set.
        title (str, optional): Title of the plot. Defaults to 'True vs Predicted Values'.
        filename (str, optional): If provided, saves the plot to the specified file path.
        hardLimit (bool, optional): If True, sets x and y limits to [-10, 5]. Defaults to True.

    The function creates a scatter plot comparing true and predicted values for both
    training and validation sets, includes a reference line (y = x), and optionally saves the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot([y_val.min(), y_val.max()], [
             y_val.min(), y_val.max()], 'r--', lw=2, alpha=0.5)
    plt.scatter(y, y_pred, alpha=0.5, s=2,
                label='Train Predictions', color='orange')
    plt.scatter(y_val, y_val_pred, alpha=0.5, s=2,
                label='Validation Predictions', color='blue')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    if hardLimit:
        plt.xlim(-10, 5)
        plt.ylim(-10, 5)
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
