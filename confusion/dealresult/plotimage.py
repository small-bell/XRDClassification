import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='混淆矩阵',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=310, size=12)
    plt.yticks(tick_marks, classes, size=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('label', size=12)
    plt.xlabel('predict', size=12)


df = pd.read_csv('all-confusion.csv')
y_true = df.iloc[:, 0].to_numpy()
y_pred = df.iloc[:, 1].to_numpy()
# y_true = [...]
# y_pred = [...]

cm = confusion_matrix(y_true, y_pred)

classes = [str(i) for i in range(10)]

# 绘制混淆矩阵图表
plot_confusion_matrix(cm, classes, normalize=True, title='Normalized Confusion Matrix')
plt.show()

