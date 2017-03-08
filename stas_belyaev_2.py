import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import functools
import operator

# metrics (assuming that we working with tuples)
minkovsky = lambda q: lambda x, y: (np.absolute(np.subtract(x, y)) ** q).sum() ** (1 / q)
euclid = minkovsky(2)
taxicab = lambda x, y: np.absolute(np.subtract(x, y)).sum()


# caching euclid for speed-up (good for small tuples)
@functools.lru_cache(maxsize=None)
def cached_euclid(xi, xj):
    @functools.lru_cache(maxsize=None)
    def _cached_euclid(xi, xj):
        return euclid(xi, xj)

    return _cached_euclid(xj, xi) if xi > xj else _cached_euclid(xi, xj)


# image preview using matplotlib
def show(image):
    plt.imshow(image, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()
    return


# step 1.1
def read_image(path):
    # read
    image = cv.imread(path)
    # transform bgr to rgb
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


# step 1.2
def get_pixels(image):
    # reshaping
    shape = image.shape
    image = image.reshape((shape[0] * shape[1], shape[2]))
    # to list of tuples
    image = list(map(lambda x: (x[0], x[1], x[2]), image.tolist()))
    return image


# one of the mu init methods
def plusplus(X, n_clusters, distance_metric):
    # initial first mu
    n = len(X)
    mu = [np.random.randint(0, n - 1)]

    # initial dists
    dists = []
    for i in range(n):
        dists.append(distance_metric(X[i], X[mu[0]]) ** 2)

    # n_clusters times we calculate new mu
    for _ in range(n_clusters - 1):
        lastmu = mu[len(mu) - 1]
        sum = 0
        for i in range(n):
            dists[i] = min(dists[i], distance_metric(X[i], X[lastmu]) ** 2)
            sum += dists[i]

        rnd = np.random.random() * sum
        sum = 0
        for i in range(n):
            if sum > rnd:
                newmu = i - 1
                break
            sum += dists[i]
            newmu = i

        mu.append(newmu)

    return list(map(lambda u: X[u], mu))


# one of the mu init method (for testing)
def presave16(X, n_clusters, distance_metric):
    return [(2, 92, 160), (254, 254, 254), (10, 11, 13), (97, 29, 27), (7, 48, 86), (73, 73, 75), (225, 181, 147),
            (145, 143, 143), (34, 32, 36), (248, 214, 11), (37, 57, 64), (175, 47, 42), (195, 42, 38), (114, 111, 110),
            (193, 141, 97), (247, 222, 104)]


# step 2
def k_means(X, n_clusters, distance_metric, mu_init=plusplus):
    # mu init
    mu = mu_init(X, n_clusters, distance_metric)

    # calc c and mu
    n = len(X)
    m = len(mu)
    k = len(X[0])
    c = list(np.random.randint(n_clusters, size=n))
    mu_stable = False
    while not mu_stable:
        # recalc c
        for i in range(n):
            c[i] = 0
            cmax = distance_metric(X[i], mu[0])
            for j in range(1, m):
                mbmax = distance_metric(X[i], mu[j])
                if mbmax < cmax:
                    c[i] = j
                    cmax = mbmax

        # precalc for new mu
        sum_mu = [tuple([0] * k)] * m
        size_mu = [0] * m
        for i in range(n):
            sum_mu[c[i]] = tuple(map(operator.add, sum_mu[c[i]], X[i]))
            size_mu[c[i]] += 1

        # recalc mu
        mu_stable = True
        for i in range(m):
            new_mu = tuple((np.asarray(sum_mu[i]) / size_mu[i]).astype(int))
            if new_mu != mu[i]:
                mu_stable = False
            mu[i] = new_mu

    return c, mu


# step 3.1
def centroid_histogram(labels):
    n_clusters = max(labels) + 1
    hist = [0] * n_clusters
    for i in range(len(labels)):
        hist[labels[i]] += 1
    return hist


# step 3.2
def plot_colors(hist, centroids):
    # init
    width, length = 50, 512
    bar = np.zeros((width, length, 3), np.uint8)

    # draw
    total = sum(hist)
    tsum = length
    start_x = 0
    for num, color in zip(hist, centroids):
        tlen = round((float(num) / total) * length) if color != centroids[len(centroids) - 1] else tsum
        tsum -= tlen
        end_x = start_x + tlen - 1
        cv.rectangle(bar, (start_x, 0), (end_x, width), np.array(color).astype("uint8").tolist(), -1)
        start_x = end_x + 1

    # return
    return bar


# step 4
def recolor(image, n_colors):
    # getting list of pixels
    pixels = get_pixels(image)

    # clustering (you can change mu_init to presave16 for skip mu init stage)
    clustering, centroids = k_means(pixels, n_colors, cached_euclid, mu_init=plusplus)

    # ploting for measure clustering results
    hist = centroid_histogram(clustering)
    bar = plot_colors(hist, centroids)

    # recoloring
    new_pixels = list(map(lambda c: centroids[c], clustering))
    new_image = np.array(new_pixels, dtype=np.uint8).reshape(image.shape)

    return bar, new_image


# step 5
def write_image(image, path):
    # transform rgb back to bgr
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    # write
    cv.imwrite(path, image)
    return


if __name__ == '__main__':
    # reading image
    image = read_image("superman-batman.png")

    # recoloring image, returning color bar and new image
    color_bar, new_image = recolor(image, 16)

    # writing bar and image
    write_image(color_bar, "color-bar.png")
    write_image(new_image, "superman-batman16.png")
