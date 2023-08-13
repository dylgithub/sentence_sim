import torch
from torch import nn, Tensor
import torch.nn.functional as F
from enum import Enum


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:

    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).

    Margin is an important hyperparameter and needs to be tuned respectively.

    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss

    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.

    """

    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN,
                 triplet_margin: float = 2):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor, rep_pos, rep_neg):
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()

if __name__ == '__main__':
    triple_loss = TripletLoss()
    anchor = torch.rand(6, 4, 3)
    pos = torch.rand(6, 4, 3)
    neg = torch.rand(6, 4, 3)
    print(triple_loss(anchor, pos, neg))