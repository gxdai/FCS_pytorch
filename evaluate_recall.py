"""
calculate top k recall.
"""
import numpy as np
from scipy.spatial import distance
import time

def evaluate_recall(feature_set, label_set, k_set):
    '''
    Args:
        feature_set:    NxD (N is sample number, D is feature size)
        label_set:      Nx1
        k_set:          return top k recall
    Returns: recall_set:     the same length as k_set.
    '''

    feature_num = feature_set.shape[0]
    label_num = label_set.shape[0]
    assert feature_num == label_num, "feature number doesn't match label number"

    distance_matrix = distance.cdist(feature_set, feature_set)

    recall_k = np.zeros(len(k_set))
    # python index starts from 0
    k_set = list(map(lambda x: x-1, k_set))

    for i in range(feature_num):
        distance_i = distance_matrix[i]
        # sort the i-th feature
        index_array = np.argsort(distance_i)

        # remove the diagonal elements
        index_array = index_array[1:]

        # Get the ranked labels based on index_array
        ranked_labels = label_set[index_array]

        # check results, 1 for correct recall, 0 for wrong recall
        ranked_results = (ranked_labels == label_set[i]).astype(int)
        recall_sum = np.cumsum(ranked_results)

        recall_k_i = (recall_sum[k_set] >= 1).astype(float)
        recall_k += recall_k_i

    recall_k = recall_k / feature_num

    return recall_k



if __name__ == '__main__':
    feature_set = np.random.random((5000, 5))
    label_set = np.reshape(np.mod(np.random.permutation(5000), 100), (-1,))
    label_set = label_set.astype(int)
    k_set = [1, 2, 4, 8]

    distM = distance.cdist(feature_set, feature_set)
    nn_av = NN.RetrievalEvaluation(distM, label_set, label_set, testMode=2)
    recall_k = evaluate_recall(feature_set, label_set, k_set)

    print(recall_k)
    print(nn_av)


