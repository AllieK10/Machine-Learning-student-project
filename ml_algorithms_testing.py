import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics as mtr
from sklearn import model_selection as ms


def Read_From_CSV(name, header=None):
    df = pd.read_csv(name, header=header)
    return df


def split_X_y(d, y_is_last=True):
    if y_is_last:
        X = d[:, :-1]
        y = d[:, -1]
    else:
        y = d[:, 0]
        X = d[:, 1:]
    return X, y


def data_preparing(d, id_column=None,
                   n_bins=3,
                   strategy='uniform',
                   missing_value=None,
                   y_is_last=True,
                   categ_cols=[]):
    if id_column is not None:
        d = d[:, 1:]

    if missing_value is not None:
        if missing_value != np.nan:
            d = np.where(d == missing_value, np.nan, d)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(d)
        d = imp.transform(d)

    if categ_cols != 'all':
        X, y = split_X_y(d, y_is_last=y_is_last)
        l = []
        for i in categ_cols:
            l.append(np.reshape(X[i], (X[i].size, 1)))
        np.delete(X, categ_cols, 1)
        d = np.hstack([X] + l + [np.reshape(y, (y.size, 1))])
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        n = len(categ_cols) + 1
        est.fit(d[:, :-n])
        d[:, :-n] = est.transform(d[:, :-n])

    enc = OneHotEncoder(drop='if_binary').fit(d)
    d = enc.transform(d).toarray()

    return split_X_y(d, y_is_last=y_is_last)


def test():
    df = Read_From_CSV('fertility_Diagnosis.txt')

    d = np.array(df)

    X, y = data_preparing(d)

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)

   # clf = Perceptron(tol=1e-3, random_state=0)
    #clf = DecisionTreeClassifier(random_state=0)
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = mtr.accuracy_score(y_test, y_pred)
    print('accuracy =', accuracy)
    precision = mtr.precision_score(y_test, y_pred, zero_division=1)
    print('precision =', precision)
    recall = mtr.recall_score(y_test, y_pred, zero_division=1)
    print('recall =', recall)
    f1 = mtr.f1_score(y_test, y_pred, zero_division=1)
    print('f1 =', f1)


if __name__ == '__main__':
    test()
