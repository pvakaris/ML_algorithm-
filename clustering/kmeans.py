import numpy as np


def recalculate_centroids(k, points, clusters):
    centroids = []
    for i in range(k):
        cluster_points = points[clusters == i]
        if cluster_points.size > 0:
            centroids.append(np.mean(cluster_points, axis=0))
        else:
            centroids.append(points[np.random.randint(len(points))])
    return np.array(centroids)

def k_means(points, clusters, limit, centroids):
    cluster_switches = 0
    for i in range(len(points)):
        distances_to_centroids = []
        for j in range(len(centroids)):
            distances_to_centroids.append(np.linalg.norm(points[i] - centroids[j]))
        smallest_value = min(distances_to_centroids)
        new_cluster_number = distances_to_centroids.index(smallest_value)
        if new_cluster_number != clusters[i]:
            cluster_switches += 1
        clusters[i] = new_cluster_number
    
    if limit <= 0 or cluster_switches == 0:
        return True
    else:
        new_centroids = recalculate_centroids(len(centroids), points, clusters)
        return k_means(points, clusters, limit - 1, new_centroids)
    
    
    
    
# Number of clusters
k = 3

# Max algorithm cycles
limit = 20

# Points in the space
points = np.array([[5, 8], [6, 7], [6, 4], [5, 7], [5, 5], [6, 5], [1, 7], [7, 5], [6, 5], [6, 7]])

# Clustes numbers. Initially all are k
clusters = np.full(len(points), k)

# Number of dimensions
n = np.ndim(points)

# Select k initial centroids randomly
initial_centroids = points[np.random.choice(points.shape[0], size=k, replace=False)]

# Run the k-means algorithm
k_means(points, clusters, limit, initial_centroids)

# Print the results
result = np.column_stack((points, clusters))
print(result)