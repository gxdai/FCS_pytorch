"""
This module is for contructing loss function.
"""
import torch


def calculate_distance_and_similariy_label(features, features_, labels, labels_, sqrt=True, pair_type='vector'):
    """
    The calculate is based on following equations
     X: (N, M)
     Y: (P, M)

     Each row represents one sample.

     the pairwise distance between X and Y is formulated as


    TO BE CONTINUED.


    Args:
        features: (N, M)
        features_: (N, M)
        labels: (N,)
        labels_: (N,)
        pair_type: str
            "vector":   generating N pairs
            "matrix":   generating N^2 pairs

    Returns:
        pairwise_distances: (N,) for "vector", (N, N) for "matrix"
        pairwise_similarity_labels:   (N,) for "vector", (N, N) for "matrix"

    """
    def get_squared_features(features):
        """
        elementwised operation.
        """
        features_l2_norm = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        return features_l2_norm
    # reshape label for convenience

    if pair_type is None or pair_type == 'matrix':

        labels = labels.view(-1, 1)
        labels_ = labels_.view(-1, 1)

        # calcualte pairwise distance
        squared_features = get_squared_features(features)
        squared_features_ = get_squared_features(features_).permute(1, 0)

        correlation_term = torch.mm(features, features_.permute(1, 0))
        pairwise_distances = squared_features + squared_features_ - 2. * correlation_term

        # calcualte pairwise similarity labels
        num_labels = labels.size(0)
        num_labels_ = labels_.size(0)
        tiled_labels = labels.repeat(1, num_labels_)
        tiled_labels_ = labels_.repeat(num_labels, 1)


        # pairwise_similarity_labels = torch.eq(tiled_labels.view(-1), tiled_labels_.view(-1)).type(torch.cuda.FloatTensor)
        pairwise_similarity_labels = torch.eq(tiled_labels.view(-1), tiled_labels_.view(-1))
        pairwise_similarity_labels = pairwise_similarity_labels.view(num_labels, num_labels_)


    elif pair_type == 'vector':

        pairwise_distances = torch.sum(torch.pow(features-features_, 2), dim=1)
        pairwise_similarity_labels = torch.eq(labels, labels_)

    if sqrt:
        # return the sqrt(distance)
        pairwise_distances = torch.sqrt(pairwise_distances + 1e-8) # To make sure the training is stable.

    return pairwise_distances, pairwise_similarity_labels.type(torch.cuda.FloatTensor)

"""
def linear_transform(x, mean_value=0.5, std_value=2.):
    y = (x - mean_value) / std_value
    return y
"""

def contrastive_loss(pairwise_distances, pairwise_similarity_labels, margin=1):
    """
    formulate constrastive loss.
    """

    # positive pair loss
    positive_pair_loss = torch.pow(pairwise_distances, 2) * pairwise_similarity_labels
    positive_pair_loss = torch.mean(positive_pair_loss)
    # negative pair loss
    negative_pair_loss = (1. - pairwise_similarity_labels) * torch.pow(torch.clamp(margin - pairwise_distances, 0.0), 2)
    negative_pair_loss = torch.mean(negative_pair_loss)

    loss = positive_pair_loss + negative_pair_loss


    return loss, positive_pair_loss, negative_pair_loss

def triplet_loss(pairwise_distances, pairwise_similarity_labels, margin=1):
    """Create the triplet loss."""
    anchor_positive_distance = torch.unsqueeze(pairwise_distances, 2)
    anchor_negative_distance = torch.unsqueeze(pairwise_distances, 1)
    triplet_loss = margin + anchor_positive_distance - anchor_negative_distance
    i_equal_j = torch.unsqueeze(pairwise_similarity_labels, 2)
    j_equal_k = torch.unsqueeze(pairwise_similarity_labels, 1)

    mask = i_equal_j * (1 - j_equal_k)
    effective_pair_num = torch.sum(mask)

    # hinge loss
    triplet_loss = torch.clamp(triplet_loss, 0.)
    # apply mask
    triplet_loss = triplet_loss * mask
    # average loss
    # triplet_loss = torch.sum(triplet_loss) / effective_pair_num
    triplet_loss = 3.0 * torch.mean(triplet_loss)

    return triplet_loss, triplet_loss, triplet_loss

def focal_triplet_loss(pairwise_distances, pairwise_similarity_labels, margin=1, mean_value=0.5, std_value=2.0):

    def linear_transform(x, mean_value=0.5, std_value=2.):
        y = (x - mean_value) / std_value
        return y



    anchor_positive_distance = torch.unsqueeze(pairwise_distances, 2)
    anchor_negative_distance = torch.unsqueeze(pairwise_distances, 1)
    triplet_loss = margin + anchor_positive_distance - anchor_negative_distance
    i_equal_j = torch.unsqueeze(pairwise_similarity_labels, 2)
    j_equal_k = torch.unsqueeze(pairwise_similarity_labels, 1)

    mask = i_equal_j * (1 - j_equal_k)
    effective_pair_num = torch.sum(mask)

    # hinge loss
    triplet_loss = torch.clamp(triplet_loss, 0.)
    # Get focal factor
    factor = 2. * torch.sigmoid(linear_transform(triplet_loss, mean_value, std_value))

    # apply mask and factor
    triplet_loss = factor * triplet_loss * mask
    # average loss
    # triplet_loss = torch.sum(triplet_loss) / effective_pair_num
    triplet_loss =  3.0 * torch.mean(triplet_loss)

    return triplet_loss, triplet_loss, triplet_loss


def focal_contrastive_loss(pairwise_distances, pairwise_similarity_labels, margin, mean_value=0.5, std_value=2.):

    def linear_transform(x, mean_value=0.5, std_value=2.):
        y = (x - mean_value) / std_value
        return y


    positive_factor = 2. * torch.sigmoid(linear_transform(pairwise_distances, mean_value, std_value))
    positive_pair_loss = torch.mean(positive_factor * torch.pow(pairwise_distances, 2) * pairwise_similarity_labels)

    negative_distances = torch.clamp(margin-pairwise_distances, 0.0)

    negative_factor = 2. * torch.sigmoid(linear_transform(negative_distances, mean_value, std_value))

    negative_pair_loss = torch.mean(negative_factor * (1. - pairwise_similarity_labels) *\
            torch.pow(negative_distances, 2))

    loss = positive_pair_loss +  negative_pair_loss

    return loss, positive_pair_loss, negative_pair_loss


if __name__ == '__main__':
    feature_1 = torch.randn(10, 5)
    label_1 = torch.randn(10)
    feature_2 = torch.randn(12, 5)
    label_2 = torch.randn(12)
    pair_dist, pair_sim_label = calculate_distance_and_similariy_label(feature_1, feature_2, label_1, label_2)
    print(pair_dist.size(), pair_sim_label.size())
    loss = contrastive_loss(pair_dist, pair_sim_label, 1.)
    print(loss.item())
