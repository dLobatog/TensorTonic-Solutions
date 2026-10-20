import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    # 1.create initial random centers
    # 2. assign all points to closest center
    # 3. measure euclidean distances to all centers
    # 4. calculate mean of these distances. This is the new center
    # 5. re- assign points to closest center
    # 6. come back to 3 until centers don't move or max_iters
    X = np.array(X) # [B, D]
    rng = np.random.RandomState(seed)
    centroids = X[rng.choice(X.shape[0], k, replace=False)] # [K, D]
    # print(centroids)
    for i in range(max_iters):
        differences = X[:, None, :] - centroids # [B, 1, D] - [K, D] = [B, K, D]
        distances = np.sum(differences ** 2, axis=2) # [B, K] (removed D)
        labels = np.argmin(distances, axis=1) # [B, K] -> [B] - for each point, 1 centroid
        new_centroids = np.zeros(centroids.shape)
        for c in range(k):
            cluster_points = X[labels==c] # [B_c, D]
            if len(cluster_points) == 0:
                new_centroids[c] = centroids[c]
                continue 
                
            new_centroids[c] = np.mean(cluster_points, axis=0) # [B_c, D] -> [D] , average point
        if np.allclose(centroids, new_centroids):
            return (labels, centroids)
        centroids = new_centroids

    differences = X[:, None, :] - centroids # [B, 1, D] - [K, D] = [B, K, D]
    distances = np.sum(differences ** 2, axis=2) # [B, K] (removed D)
    labels = np.argmin(distances, axis=1) 
    return (labels, centroids)