"""Evaluate the peformance."""
import numpy as np
from scipy.spatial import distance
import cub_2011
import RetrievalEvaluation
import evaluate_recall
import evaluate_clustering
import torch
import copy

def get_feature_and_label(model, dataloader, device):
    model.eval()
    feature_set = []
    label_set = []
    path_set = []
    # no need for gradients calculation.
    with torch.no_grad():
        for data in dataloader:
            img, label, path = data['img'].to(device), data['label'].to(device), data['path']
            # output = model.forward_once(img)
            output, output_2 = model(img, img)
            feature_set.append(output.detach())
            label_set.append(label.detach())
            path_set.extend(path)

    feature_set = torch.cat(feature_set, dim=0)
    label_set = torch.cat(label_set)

    return feature_set.cpu().numpy(), label_set.cpu().numpy(), path_set


def evaluation(feature_set, label_set, k_set=[1, 2, 4, 8, 16, 32]):
    distM = distance.cdist(feature_set, feature_set)
    nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation.RetrievalEvaluation(distM, label_set, label_set, testMode=2)
    recall_k = evaluate_recall.evaluate_recall(feature_set, label_set, k_set)

    for i, k in enumerate(k_set):
        print("Recall@{:1d}: {:6.5f}".format(k_set[i], recall_k[i]))

    normalized_mutual_information, RI, F = evaluate_clustering.evaluate_clustering(feature_set, label_set)
    print("")
    print("normalized_mutual_information = {}".format(normalized_mutual_information))
    print("RI = {}".format(RI))
    print("F_1 = {}".format(F))
    print("")


    print(('The NN is {:5.5f}\nThe FT is {:5.5f}\n' +
           'The ST is {:5.5f}\nThe DCG is {:5.5f}\n' +
           'The E is {:5.5f}\nThe MAP {:5.5f}\n').format(
           nn_av, ft_av, st_av, dcg_av, e_av, map_))

    rec = np.expand_dims(rec, axis=1)
    pre = np.expand_dims(pre, axis=1)

    rec_pre = np.concatenate([rec, pre], axis=1)
    return rec_pre
