import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def cluster_centers(data, n_clusters):
    centers_idxs = []
    data_new = data.copy()
    for i in range(n_clusters):
        dist_matrix = euclidean_distances(data_new, data_new)
        c_idx = dist_matrix.sum(axis=1).argsort()[::-1][0]
        centers_idxs.append(c_idx)
        data_new = np.delete(data_new, c_idx, axis=0)

    return euclidean_distances(data, data), np.array(centers_idxs)

def splayed_centers(data, n_clusters):
    dist_matrix = euclidean_distances(data, data)
    max_dist_idx = np.argmax(dist_matrix)
    idx1, idx2 = np.unravel_index([max_dist_idx], dist_matrix.shape)
    centers_idxs = [idx1, idx2]
    for i in range(n_clusters - 2):
        points_dists = np.zeros(data.shape[0])
        for idx, point in enumerate(data):
            if idx not in centers_idxs:
                points_dists[idx] = np.min([dist_matrix[idx, c_idx] for c_idx in centers_idxs])
        centers_idxs.append(np.argmax(points_dists))
    return dist_matrix, centers_idxs

def find_labels(method, n_clusters, data):
    if method == 'KMeans':
        labels = KMeans(n_clusters=n_clusters).fit_predict(data)
    elif method == 'MaxDistance':
        labels = []
        dist_matrix, centers_idxs = cluster_centers(data, n_clusters)
        for idx, point in enumerate(data):
            labels.append(np.argmin([dist_matrix[idx, c_idx] for c_idx in centers_idxs]))
    elif method == 'Splayed':
        labels = []
        dist_matrix, centers_idxs = splayed_centers(data, n_clusters)
        for idx, point in enumerate(data):
            labels.append(np.argmin([dist_matrix[idx, c_idx] for c_idx in centers_idxs]))
    return np.array(labels)

def fill_phi_splayed_method():
    dist_matrix = euclidean_distances(data, data)
    idx1, idx2 = np.argmax(dist_matrix)
       
    for i, point in enumerate(data):
        if  i not in [idx1, idx2]:
            np.argmin([dist_matrix[i, idx1], dist_matrix[i, idx2]])

    return phi

def initialize_phi(
         method, 
         n_topics,
         n_topics_health,
         file_data,
         file_true_labels):
    data = np.load(file_data)
    y_true = np.load(file_true_labels)

    k = 216
    phi = np.zeros((k + 2, n_topics))

    ind_health = y_true == 0
    data_health = data[ind_health]
    data_dis = data[~ind_health]

    labels = find_labels(method, n_topics_health, data_health)
    for n_topic in range(n_topics_health):
        phi[:k, n_topic] = np.sum(data_health[labels == n_topic], axis=0)

    labels = find_labels(method, n_topics - n_topics_health, data_dis)
    for n_topic in range(n_topics_health, n_topics):
        phi[:k, n_topic] = np.sum(data_dis[labels == n_topic - n_topics_health], axis=0)

    phi[k, :] = np.ones(n_topics) * (np.count_nonzero(ind_health))
    phi[k + 1, :] = np.ones(n_topics) * (y_true.shape[0] - phi[k, :][0])
    return phi

