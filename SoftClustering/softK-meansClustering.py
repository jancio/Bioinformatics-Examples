# 03.04.2017					#
# Jan Ondras					#
# Soft/Fuzzy k-means clustering	#
#################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Euclidian distance as distance metric
def distance(d, c):
	return np.linalg.norm(d - c)


# GENERATE TEST DATA - datapoints
K = 4		# K = number of cluster
N = 1000 	# N = total number of datapoints
x_data = np.zeros(0)
y_data = np.zeros(0)
test_means = [[0.4, 0.2], [0.3, 0.7], [0.1, 0.3], [0.5, 0.6]]
test_stds = [[0.08, 0.03], [0.03, 0.05], [0.02, 0.15], [0.02, 0.02]]
for m, s in zip(test_means, test_stds):
	x_data = np.hstack((x_data, np.random.standard_normal(N/K)*s[0] + m[0]))
	y_data = np.hstack((y_data, np.random.standard_normal(N/K)*s[1] + m[1]))
datapoints = np.array([x_data, y_data]).T
plt.subplot(3, 1, 1)
plt.plot(datapoints[:,0], datapoints[:,1], 'o')
plt.title('Initial data: ' + str(K) + ' clusters, ' + str(N/K) + ' points each')


# INITIALISATION
max_iter = 100						# Terminates if (totalIterations >= max_iter) OR (change <= tolerance)
tolerance = 1e-5
totalIterations = 0
change = np.inf
changes = []
m = 5 								# Fuzzyness coefficient
centers = np.zeros((K, 2))			# (K x 2) init. positions of centers
									# (N x K) Partition matrix W: w_ij is the probability of datapoint x_i belonging to cluster c_j
W = np.random.random_sample((N, K))	# Randomly initialize W
W = normalize(W, axis=1, norm='l1')	# Normalize: rows sum to one


while (totalIterations < max_iter) and (change > tolerance):
	totalIterations += 1

	# 1.) Calculate new centers as weighted average of datapoints; weights are given by membership probabilities - from W
	for j, clusterProbs in enumerate(W.T):
		# check div by zero, all weights for cluster are zero
		if np.sum(clusterProbs) != 0.0:
			centers[j] = np.average(datapoints, weights=np.power(clusterProbs, m), axis=0)
		else:
			print "Cluster " + str(j) + " is temporarily empty."

	# 2.) Update partition matrix W
	change = 0.0					# to calculate maximal change in W (elementwise); for termination criterion
	for i in range(N):
		# Iterate over all clusters (j) for each datapoint (i)
		for j in range(K):
			old_w = W[i, j]
			power = 2.0/(1-m)
			W[i, j] = np.power(distance(datapoints[i], centers[j]), power) / np.sum( [pow(distance(datapoints[i], c), power) for c in centers] )

			if abs(W[i, j] - old_w) > change:
				change = abs(W[i, j] - old_w)

	print "Iteration " + str(totalIterations) + " 	change: " + str(change)
	changes.append(change)

# For each datapoint choose most probable cluster: save corresponding cluster index only
dataAssignmentToClusters = [np.argmax(row) for row in W]


# Visualise maximal change in membership probabilities vs. iteration
plt.subplot(3, 1, 2)
plt.plot(range(1, totalIterations + 1), changes)
plt.title('Maximal change in membership probabilities vs. iteration')
plt.xlabel('Iteration')
plt.ylabel('Maximal change in membership probabilities')

# Visualise final clusters
plt.subplot(3, 1, 3)
plt.scatter(centers[:,0], centers[:,1], s=1000, lw=3, c='black', marker='+', zorder=2)
for j in range(K):
	dataIndicesForCluster_j = [i for i, c in enumerate(dataAssignmentToClusters) if c == j]
	if len(dataIndicesForCluster_j) != 0:
		plt.plot(datapoints[dataIndicesForCluster_j,0], datapoints[dataIndicesForCluster_j,1], 'o', zorder=1)
	else:
		print "Cluster " + str(j) + " is empty."
plt.title('Clustered data: ' + str(K) + ' clusters, ' + str(N/K) + ' points each, #iterations = ' + str(totalIterations) + ', m = ' + str(m))
plt.show()

# Time complexity:	O(totalIterations * N * K * K) - updating the partition matrix dominates in time complexity
# Space complexity:	O(N * K) - partition matrix