# 14.12.2016			#
# Jan Ondras			#
# k-means clustering	#
#########################

import numpy as np
import matplotlib.pyplot as plt

# Euclidian distance as distance metric
def distance(d, c):
	return np.linalg.norm(d - c)

# K = number of cluster
# N = number of datapoints
# D-dimensional data
# Max number of iterations
# Tolerance
# Terminates if (totalIterations >= max_iter) OR (change <= tolerance)
K = 7
N = 1000
D = 2
max_iter = 100000
tolerance = 1e-5
totalIterations = 0
change = np.inf
changes = []

# N x D, datapoints taken randomly from interval [0,1]
datapoints = np.random.random_sample((N, D))
# K x D, random init. position of centers within interval [0,1]
centers = np.random.random_sample((K, D))

# Visualise initial data
plt.subplot(3, 1, 1)
plt.plot(datapoints[:,0], datapoints[:,1], 'o')
plt.scatter(centers[:,0], centers[:,1], s=500, c='red', marker='+')
plt.title('Initial data')
plt.axis([0, 1, 0, 1])

while (totalIterations < max_iter) and (change > tolerance):
	totalIterations += 1

	# K x (variable) size of cluster
	# associate only indices of datapoints with clusters => lower space requirements
	clusters = [[] for i in range(K)]

	# Assign datapoints to clusters(centers)
	for i, d in enumerate(datapoints):
		minDist = np.inf
		for j, c in enumerate(centers):
			dist = distance(d, c)
			if minDist > dist:
				minDist = dist
				minJ = j
		# associate datapoint's index with closest center
		clusters[minJ].append(i)

	# Recompute new centers
	old_centers = centers.copy()
	for i, cl in enumerate(clusters):
		# If cluster is non-empty, then calculate new Center of Mass, else leave the previous center
		if(len(cl) != 0):
			centers[i] = np.mean(datapoints[np.asarray(cl)], axis=0)

	# Change in centers positions at this iteration
	change = np.max(np.linalg.norm(centers - old_centers, axis=1))			# maximum of changes in centers positions
	#change = np.sum(np.linalg.norm(centers - old_centers, axis=1))			# (alternatively) sum of changes in centers positions
	#change = np.mean(np.linalg.norm(centers - old_centers, axis=1))		# (alternatively) average of changes in centers positions
	print "Iteration " + str(totalIterations) + " 	change: " + str(change)
	changes.append(change)

# Visualise change in centers positions vs. iteration
plt.subplot(3, 1, 2)
plt.plot(range(1, totalIterations + 1), changes)
plt.title('Change in centers positions vs. iteration')

# Visualise final clusters
plt.subplot(3, 1, 3)
for i, cl in enumerate(clusters):
	cl_arr = np.asarray(cl)
	if(cl_arr.size == 0):
		print "Empty cluster with center at: " + str(centers[i])
	else:
		plt.plot(datapoints[cl_arr,0], datapoints[cl_arr,1], 'o')
plt.scatter(centers[:,0], centers[:,1], s=500, c='red', marker='+')
plt.title('Clustered data: K = ' + str(K) + ', N = ' + str(N) + ', iterations = ' + str(totalIterations))
plt.axis([0, 1, 0, 1])
plt.show()

# Time complexity:	O(totalIterations * N * K * D)
# Space complexity:	O(N * D)