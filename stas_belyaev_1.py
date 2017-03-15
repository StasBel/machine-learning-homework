import numpy as np
import pandas as pd
from collections import Counter

# Данные
data_file = "wine.csv"
names = list(map(lambda s: s[6:-1], open(data_file).readlines()[:14]))


# Чтение csv файла в DataFrame
def read_data():
    df = pd.read_csv(data_file, skiprows=15, header=None, names=names)
    y = df[names[0]]
    X = df.drop(names[0], 1)
    return X, y


# Задание 1
def train_test_split(X, y, ratio):
    indicies = X.sample(frac=ratio).index.sort_values()
    X_train = X.loc[indicies].reset_index(drop=True)
    y_train = y.loc[indicies].reset_index(drop=True)
    X_test = X.drop(indicies).reset_index(drop=True)
    y_test = y.drop(indicies).reset_index(drop=True)
    return X_train, y_train, X_test, y_test


# Метрики
minkovsky = lambda q: lambda x, y: ((x - y) ** q).sum() ** (1 / q)
euclid = minkovsky(2)
taxicab = lambda x, y: (x - y).abs().sum()


# Аналог функции a(u, X^l) из лекции
def a(X_train, y_train, k, dist):
    # Функция для подсчета самого частого значения в листе интов
    most_popular = lambda l: Counter(l).most_common(1)[0][0]

    return lambda u: most_popular(
        X_train.apply(lambda row: dist(u, row), axis=1)
            .nsmallest(k)
            .index
            .map(lambda ind: y_train.loc[ind])
            .tolist()
    )


# Задание 2
def knn(X_train, y_train, X_test, k, dist):
    return X_test.apply(a(X_train, y_train, k, dist), axis=1)


# Задание 3
def precision_recall(y_pred, y_test, n_class):
    tp = ((y_pred == n_class) & (y_test == n_class)).sum()
    fp = ((y_pred == n_class) & (y_test != n_class)).sum()
    fn = ((y_pred != n_class) & (y_test == n_class)).sum()
    return tp / (tp + fp), tp / (tp + fn)


def print_precision_recall(y_pred, y_test, print_info=None):
    if (print_info):
        print(print_info)

    n_classes = np.unique(y_pred).tolist()
    for n_class in n_classes:
        p, r = precision_recall(y_pred, y_test, n_class)
        print("Class: ", n_class, "\t Precision: ", p, "\t Recall: ", r)


# Задание 4
def loocv(X_train, y_train, dist):
    # loo
    def loo(k):
        a_calc = X_train.apply(lambda row: a(X_train.drop(row.name), y_train.drop(row.name), k, dist)(row), axis=1)
        return (a_calc != y_train).sum()

    cand_list = np.arange(1, len(X_train) - 1)
    cands = pd.Series(data=cand_list, index=cand_list)
    opt_k = cands.map(lambda cand: loo(cand)).idxmin()

    return opt_k


# Задание 5: Нет, не правда
def test(X_train, y_train, X_test, y_test, methods):
    for k, dist, print_info in methods:
        y_pred = knn(X_train, y_train, X_test, k, dist)
        print_precision_recall(y_pred, y_test, print_info)
    return


if __name__ == '__main__':
    # Читаем данные
    X, y = read_data()

    # Разделяем выборки
    X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=0.5)

    # Массив методов; метод в сущности отличается функцией расстояния; loocv можно пересчитать, сняв комментарии
    methods = [
        (
            17,  # loocv(X_train, y_train, dist=euclid),
            euclid,
            "Using Euclid dist:"
        ),
        (
            16,  # loocv(X_train, y_train, dist=taxicab),
            taxicab,
            "Using Taxicab dist:"
        )
    ]

    # Тестируем
    test(X_train, y_train, X_test, y_test, methods)
