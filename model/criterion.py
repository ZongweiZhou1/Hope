import numpy as np
from scipy.optimize import linear_sum_assignment
from model.loss import euclidean_dist

def assignment(cost_mat, thresh=0.5):
    """
    :param cost_mat:        M x N, numpy.array
    :param thresh:          scalar
    :return:
    """
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    assign_values = cost_mat[row_ind, col_ind]
    row_ind = row_ind[assign_values<thresh]
    col_ind = col_ind[assign_values<thresh]
    return row_ind, col_ind

def evaluation(x, y, label_x, label_y, thresh=0.5):
    """
    :param x:               torch.tensor, M x F
    :param y:               torch.tensor, N x F
    :param label_x:         torch.tensor, M
    :param label_y:         torch.tensor, N
    :param thresh:          scalar
    :return:
        f_measure:          scalar, evaluation about the points
        graph_match:        {0, 1}, evaluation about the graphs
    """
    cost_mat = euclidean_dist(x, y)
    cost_mat = cost_mat.cpu().detach().numpy()
    label_x = label_x.cpu().detach().numpy()
    label_y = label_y.cpu().detach().numpy()

    row_ind, col_ind = assignment(cost_mat, thresh)
    match_tuple = [(row, col) for row, col in zip(row_ind, col_ind)]
    label_tuple = [(row, col) for row, col in \
                   zip(*np.where(label_x[:, np.newaxis] == label_y[np.newaxis, :]))]
    tp = 0
    for t in label_tuple:
        if t in match_tuple:
            tp += 1
    pre = tp/len(match_tuple) if len(match_tuple)>0 else 1.0
    rec = tp/len(label_tuple) if len(label_tuple)>0 else 1.0
    f_meature = 2 * pre * rec/(pre + rec + 1e-12)
    if f_meature > 0.95:
        graph_match = 1
    else:
        graph_match = 0
    return f_meature, graph_match
