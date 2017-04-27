from matplotlib.mlab import find
import numpy as np
import matplotlib.pyplot as plt


def visualize(clf, X, Y, axis):
    clf.fit(X, Y)

    border = .5
    h = .02

    x_min, x_max = X[:, 0].min() - border, X[:, 0].max() + border
    y_min, y_max = X[:, 1].min() - border, X[:, 1].max() + border

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    z_class = clf.predict(mesh).reshape(xx.shape)

    # Put the result into a color plot
    # fig = plt.figure(figsize=(8, 6))
    axis.pcolormesh(xx, yy, z_class, cmap=plt.cm.summer, alpha=0.3)

    # Plot hyperplane and margin
    z_dist = clf.decision_function(mesh).reshape(xx.shape)
    axis.contour(xx, yy, z_dist, [0.0], colors='black')
    axis.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

    # Plot also the training points
    Y_pred = clf.predict(X)

    ind_support = clf.ind_support
    ind_correct = list(set(find(Y == Y_pred)) - set(ind_support))
    ind_incorrect = list(set(find(Y != Y_pred)) - set(ind_support))

    axis.scatter(X[ind_correct, 0], X[ind_correct, 1], c=Y[ind_correct], cmap=plt.cm.summer, alpha=0.9)
    axis.scatter(X[ind_incorrect, 0], X[ind_incorrect, 1], c=Y[ind_incorrect], cmap=plt.cm.summer, alpha=0.9,
                 marker='*',
                 s=50)
    axis.scatter(X[ind_support, 0], X[ind_support, 1], c=Y[ind_support], cmap=plt.cm.summer, alpha=0.9, linewidths=1.8,
                 s=40)

    axis.set_xlim(xx.min(), xx.max())
    axis.set_ylim(yy.min(), yy.max())
    axis.set_title("clf: {}, C: {}, kernel?: {}".format(clf.__class__.__name__, clf.C,
                                                        clf.kernel.__name__ if hasattr(clf, "kernel") else "None"))
