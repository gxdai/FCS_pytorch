"""
This is the module evaluate the clustering performance.
"""

from sklearn.cluster import KMeans
import numpy as np
from scipy.special import comb

def get_cluster(feature_set, n_clusters=2):
    """"
    Args:
        feature_set: (N, D), N is sample number, D is feature size.
        n_clusters: int, number of clusters to be generated.

    Args:
        kmeams
    """
    # print("In side function n_clusters = {}".format(n_clusters))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_set)

    return kmeans


def get_label_statics(label_set):
    unique_labels = np.unique(label_set)
    # get sample number belong to each class
    class_num = unique_labels.shape[0]
    sample_num_per_class = np.zeros(class_num)
    for i in range(class_num):
        index = np.where(label_set == unique_labels[i])
        sample_num_per_class[i] = np.prod(index[0].shape)

    return unique_labels, sample_num_per_class, class_num





def evaluate_clustering(feature_set, label_set, n_clusters=None):
    """
    evalute the cluserting performance.
    """
    total_num = label_set.shape[0]
    assert feature_set.shape[0] == label_set.shape[0], \
            "the number of feature doesn't match the number of labels"

    unique_labels, sample_num_per_class, class_num = \
            get_label_statics(label_set)


    if n_clusters is None:
        n_clusters = class_num

    kmeans = get_cluster(feature_set=feature_set, n_clusters=n_clusters)
    cluster_set = kmeans.labels_
    # count number of samples for each cluster
    unique_clusters, sample_num_per_cluster, cluster_num_ = \
            get_label_statics(cluster_set)

    assert n_clusters == cluster_num_, "cluster number doesn't match"



    # print('sampler_num_per_class = {}'.format(sample_num_per_class))
    # print('sampler_num_per_cluster = {}'.format(sample_num_per_cluster))

    # count purity
    purity = 0.
    for i in range(n_clusters):
        # get samples belong to i-the cluster
        index = np.where(cluster_set == unique_clusters[i])
        labels = label_set[index]

        counter = np.zeros(class_num)
        for j in range(class_num):
            index_j = np.where(labels == unique_labels[j])
            counter[j] = index_j[0].shape[0]

        purity += np.amax(counter)


    purity /= float(total_num)
    # print("Purity is {:5.3f}".format(purity))

    # compute normalized mutual information

    count_cross = np.zeros((n_clusters, class_num))
    for i in range(n_clusters):
        index_i = np.where(cluster_set == unique_clusters[i])
        labels_for_cluster_i = label_set[index_i]
        for j in range(class_num):
            index_j = np.where(labels_for_cluster_i == unique_labels[j])
            count_cross[i, j] = index_j[0].shape[0]

    # print("count_cross = {}".format(count_cross))
    # mutual information
    I = 0
    for i in range(n_clusters):
        for j in range(class_num):
            if count_cross[i,j] > 0:
                s = count_cross[i,j] / total_num * np.log(total_num * count_cross[i,j] /
                    (sample_num_per_cluster[i] * sample_num_per_class[j]))
                I += s


    # print("Mutual information is {:5.5f}".format(I))

    h_cluster = 0
    for i in range(n_clusters):
        s = -sample_num_per_cluster[i] / total_num * \
            np.log(sample_num_per_cluster[i] / total_num)

        h_cluster += s

    # print("Entropy cluster is {:5.5f}".format(h_cluster))


    h_class = 0
    for j in range(class_num):
        s = -sample_num_per_class[j] / total_num * \
            np.log(sample_num_per_class[j] / total_num)

        h_class += s

    # print("Entropy class is {:5.5f}".format(h_class))

    normalized_mutual_information = 2 * I / (h_cluster + h_class)
    """
    print("normalized_mutual_information is {:5.5f}".\
        format(normalized_mutual_information))
    """
    tp_and_fp = 0
    for i in range(n_clusters):
        if sample_num_per_cluster[i] > 1:
            tp_and_fp += comb(sample_num_per_cluster[i], 2)

    # print("tp_and_fp = {}".format(tp_and_fp))
    tp = 0
    for i in range(n_clusters):
        for j in range(class_num):
            if count_cross[i,j] > 1:
                tp += comb(count_cross[i,j], 2)

    # print("tp = {}".format(tp))

    fp = tp_and_fp - tp

    # print("fp is {}".format(fp))

    count = 0
    for j in range(class_num):
        if sample_num_per_class[j] > 1:
            count += comb(sample_num_per_class[j], 2)


    fn = count - tp
    # print("fn is {}".format(fn))

    tn = comb(total_num, 2) - tp - fp -fn

    RI = (tp + tn) / comb(total_num, 2)
    # print("RI is {}".format(RI))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)


    # print("Precision is {}".format(precision))
    # print("Recall is {}".format(recall))

    beta = 1
    F = (beta * beta + 1) * precision * recall / \
        (beta * beta * precision + recall)

    # print("F_{} is {}".format(beta, F))

    return normalized_mutual_information, RI, F



if __name__ == '__main__':
    feature_set = np.random.random((5000, 100))
    label_set = np.random.permutation(5000) % 10
    NMI, RI, F = evaluate_clustering(feature_set, label_set)
    print(NMI, RI, F)


