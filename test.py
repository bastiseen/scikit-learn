from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, recall_score
from imblearn.metrics import specificity_score
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics._classification import specificity_support
from sklearn.metrics._classification import balanced_matthews_corrcoeff


y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0])

MCM = multilabel_confusion_matrix(y_true, y_pred)
print(MCM)

print("macro = ", specificity_support(y_true, y_pred, average='macro'))
print("micro = ", specificity_support(y_true, y_pred, average='micro'))
print("weighted = ", specificity_support(y_true, y_pred, average='weighted'))

mcc = balanced_matthews_corrcoeff(y_true, y_pred)
print(mcc)
