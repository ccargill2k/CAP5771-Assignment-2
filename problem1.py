# Import libraries
import numpy as np
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Defines for script run1.py
def run_problem1():

    # Load in the data
    X_train = np.loadtxt('X_train.txt')
    y_train = np.loadtxt('y_train.txt')
    X_test = np.loadtxt('X_test.txt').T
    y_test = np.loadtxt('y_test.txt')

    num_classes = y_train.shape[1]
    num_test = X_test.shape[0]

    # Polynomial Kernel
    poly_pred = np.zeros((num_test, num_classes))
    for class_idx in range(num_classes):
        y_class = y_train[:, class_idx].astype(int)
        svc = SVC(kernel='poly', degree=2)
        svc.fit(X_train, y_class)
        pred = svc.predict(X_test)
        poly_pred[:, class_idx] = pred

    # Accuracy Calc for Polynomial Kernel
    accs = []
    for i in range(num_test):
        true_labels = y_test[i]
        pred_labels = poly_pred[i]
        intersection = np.sum(np.logical_and(true_labels == 1, pred_labels == 1))
        union = np.sum(np.logical_or(true_labels == 1, pred_labels == 1))
        if union == 0:
            acc = 1.0 if intersection == 0 else 0.0
        else:
            acc = intersection / union
        accs.append(acc)
    poly_acc = np.mean(accs)

    # Print Polynomial Result
    print(f"Polynomial Kernel: {poly_acc * 100:.2f}%")

    # Gaussian Kernel
    rbf_pred = np.zeros((num_test, num_classes))
    for class_idx in range(num_classes):
        y_class = y_train[:, class_idx].astype(int)
        svc = SVC(kernel='rbf', degree=2)
        svc.fit(X_train, y_class)
        pred = svc.predict(X_test)
        rbf_pred[:, class_idx] = pred

    # Accuracy Calc for Gaussian Kernel
    accs = []
    for i in range(num_test):
        true_labels = y_test[i]
        pred_labels = rbf_pred[i]
        intersection = np.sum(np.logical_and(true_labels == 1, pred_labels == 1))
        union = np.sum(np.logical_or(true_labels == 1, pred_labels == 1))
        if union == 0:
            acc = 1.0 if intersection == 0 else 0.0
        else:
            acc = intersection / union
        accs.append(acc)
    rbf_acc = np.mean(accs)

    # Print Gaussian Results
    print(f"Gaussian Kernel: {rbf_acc * 100:.2f}%")