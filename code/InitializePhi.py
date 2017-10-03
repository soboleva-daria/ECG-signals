import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import string

def cluster_centers(data, n_clusters):
    centers_idxs = []
    data_new = data.copy()
    for i in range(n_clusters):
        dist_matrix = euclidean_distances(data_new, data_new)
        c_idx = dist_matrix.sum(axis=1).argsort()[::-1][0]
        centers_idxs.append(c_idx)
        data_new = np.delete(data_new, c_idx, axis=0)

    return euclidean_distances(data, data), np.array(centers_idxs)

def spread_centers(data, n_clusters):
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

def find_harmony_coeff(data):
    chars_idxs = pd.read_csv('chars_idxs.csv', index_col=0)      
    coeff1 = np.zeros((data.shape[0], ))
    for ch in ['A', 'B', 'E', 'F']:
        idxs, counts = np.unique(chars_idxs[ch], return_counts=True)
        coeff1 += (data[:, idxs] * counts).sum(axis=1)

    coeff2 = np.zeros((data.shape[0], ))
    for ch in ['C', 'D']:
        idxs, counts = np.unique(chars_idxs[ch], return_counts=True)
        coeff2 += (data[:, idxs] * counts).sum(axis=1)

    return np.nan_to_num(coeff1 / coeff2)

def find_labels(method, n_clusters, data):
    if method == 'KMeans':
        labels = KMeans(n_clusters=n_clusters).fit_predict(data)
    elif method == 'NaiveKMeans':
        labels = []
        dist_matrix, centers_idxs = cluster_centers(data, n_clusters)
        for idx, point in enumerate(data):
            labels.append(np.argmin([dist_matrix[idx, c_idx] for c_idx in centers_idxs]))
    elif method == 'Spread':
        labels = []
        dist_matrix, centers_idxs = spread_centers(data, n_clusters)
        for idx, point in enumerate(data):
            labels.append(np.argmin([dist_matrix[idx, c_idx] for c_idx in centers_idxs]))

    elif method == 'KMeansGram3':
        labels = KMeans(n_clusters=n_clusters).fit_predict(data.T)

    elif method == 'HarmonyBaskets':
        coeff = find_harmony_coeff(data)
        labels = KMeans(n_clusters=n_clusters).fit_predict(coeff[:, np.newaxis])
    else:
        raise Exception('Method not recognized')
    return np.array(labels)

def kmeans_gram3(data, labels, n_topic):
     freq_gram3 = np.zeros(labels.shape)
     idxs = labels == n_topic
     freq_gram3[idxs] = np.sum(data.T[idxs], axis=1)
     freq_gram3[~idxs] = np.random.uniform(
                               low=0.0,
                               high=np.min(freq_gram3[idxs]),
                               size=np.count_nonzero(~idxs))

     return freq_gram3.astype(int)

def initialize_phi(
         methods, 
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
    
    for method in methods:
        if method == 'Random':
            phi[:k, :] += np.random.uniform(low=0.0, high=1.0, size=n_topics)
            continue
        labels = find_labels(method, n_topics_health, data_health)
        for n_topic in range(n_topics_health):
            if method == 'KMeansGram3':
                freq = kmeans_gram3(data_health, labels, n_topic).copy()
            else:
                freq = np.mean(data_health[labels == n_topic], axis=0)
            phi[:k, n_topic] += freq
            
        labels = find_labels(method, n_topics - n_topics_health, data_dis)
        for n_topic in range(n_topics_health, n_topics):
            if method == 'KMeansGram3':
                freq = kmeans_gram3(data_dis, labels, n_topic - n_topics_health).copy()
            else:
                freq = np.mean(data_dis[labels == n_topic - n_topics_health], axis=0)
            phi[:k, n_topic] += freq

    phi[k, :] = np.ones(n_topics) * (np.count_nonzero(ind_health))
    phi[k + 1, :] = np.ones(n_topics) * (y_true.shape[0] - phi[k, :][0])
    return phi
