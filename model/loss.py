from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable


class TripletLoss(object):
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


def normalize(x, axis=-1):
    """ Normalizing to unit length along the specified dimension.
    :param x:
    :param axis:
    :return:
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """Euclidean distance"""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat:       pytorch Variable, pair wise distance between samples, shape [N, N]
        labels:         pytorch LongTensor, with shape [N]
        return_inds:    whether to return the indices. Save time if `False`(?)
    Returns:
        dist_ap:        pytorch Variable, distance(anchor, positive); shape [N]
        dist_an:        pytorch Variable, distance(anchor, negative); shape [N]
        p_inds:         pytorch LongTensor, with shape [N];
                        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds:         pytorch LongTensor, with shape [N];
                        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    """
    assert dist_mat.ndimension() == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    tmp = Variable(dist_mat.data.new().resize_as_(dist_mat.data).fill_(1e4))

    dist_ap, p_inds = torch.max(dist_mat - is_neg.float() * tmp, 1, keepdim=False)
    dist_an, n_inds = torch.min(dist_mat + is_pos.float() * tmp, 1, keepdim=False)
    if return_inds:
        return dist_ap, dist_an, p_inds, n_inds
    return dist_ap, dist_an


def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        global_feat: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the
          Channel dimension
    Returns:
        loss: pytorch Variable, with shape [1]
        p_inds: pytorch LongTensor, with shape [N];
          indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
          indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        ==================
        For Debugging, etc
        ==================
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat


if __name__ == '__main__':
    x = torch.rand(8, 128)
    triplet_loss = TripletLoss(margin=1)
    labels = torch.tensor([-1, 0, 1, 1, 0, 2, 2, -2]).long()
    loss, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(triplet_loss, x, labels)
    print(loss)
    print(p_inds)
    print(n_inds)
    print(dist_ap)
    print(dist_an)
    print(dist_mat)
