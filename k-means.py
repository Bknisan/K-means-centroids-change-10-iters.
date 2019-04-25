import copy
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
from scipy.misc import imread
from scipy.spatial.distance import cdist
from init_centroids import init_centroids


# get the distance between 2 3d points

def distance(first, second):
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 + (first[2] - second[2]) ** 2


# print the i iteration
def print_centroids_iter(X, iter):
    print("iter",'{0}: '.format(iter),end='')
    Y = np.asarray(X)
    if len(Y.shape) == 1:
        print(' '.join(str(np.floor(100*Y)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', '))
    else:
        print(' '.join(str(np.floor(100*Y)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1])

# calculating the closets centroid
def closest_distance(pixel, centroids_ar, avg_centroids):
    closest = centroids_ar[0]
    for centroid in centroids_ar:
        if distance(closest, pixel) > distance(centroid, pixel):
            closest = centroid
    for i in range(0, len(centroids_ar)):
        if ((closest[0] == centroids_ar[i][0]) and (closest[1] == centroids_ar[i][1]) and (
                closest[2] == centroids_ar[i][2])):
            avg_centroids[i][0] += pixel[0]
            avg_centroids[i][1] += pixel[1]
            avg_centroids[i][2] += pixel[2]
            avg_centroids[i][3] += 1
    return avg_centroids


# initialize avg of centroids.
def initialize_avg(number_of_centroids):
    if number_of_centroids == 2:
        return [[0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0]]
    elif number_of_centroids == 4:
        return [[0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0]]
    elif number_of_centroids == 8:
        return [[0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0],
                [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0]]
    else:
        return [[0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0],
                [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0],
                [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0],
                [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0], [0.0, 0.0, 0.0, 0]]


# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'
A = imread(path)
A = A.astype(float) / 255.
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], img_size[2])
centroids_number = 2
# powering num of centroids by 2 each iteration.
while centroids_number < 32:
    print("k={0}:".format(centroids_number))
    # 2d array of centroids.
    centroids = init_centroids(X, centroids_number)
    avg = initialize_avg(centroids_number)
    for i in range(0, 11):
        print_centroids_iter(centroids, i)
        for j in X:
            avg = closest_distance(j, centroids, avg)
        for k in range(0, centroids_number):
            avg[k][0] = avg[k][0] / avg[k][3]
            avg[k][1] = avg[k][1] / avg[k][3]
            avg[k][2] = avg[k][2] / avg[k][3]
            centroids[k][0] = avg[k][0]
            centroids[k][1] = avg[k][1]
            centroids[k][2] = avg[k][2]
        avg = initialize_avg(centroids_number)
    centroids_number *= 2
