import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset.graph_dataset import graph_dataset
import os.path as osp


def visualize_graphs(visual_data, points_data, record_key, with_edge=False):
    """
    :param visual_data:         torch_geometric.data
    :param points_data:         torch_geometric.data
    :param record_key:          [dataset_name, frameId]
    :return:
    """
    dataset = record_key[0]
    frameId = record_key[1]
    if dataset == 'zara01':
        dirpath = 'data/UCY/zara/zara01/'
        interval_frame = 8
    elif dataset == 'zara02':
        dirpath = 'data/UCY/zara/zara02/'
        interval_frame = 12
    elif dataset == 'univ':
        dirpath = 'data/UCY/univ'
        interval_frame = 4
    elif dataset == 'eth':
        dirpath = 'data/ETH/seq_eth'
        interval_frame = 1
    elif dataset == 'hotel':
        dirpath = 'data/ETH/seq_hotel'
        interval_frame = 1
    else:
        raise Exception('No dataset: `{}` found!'.format(dataset))
    img_path = osp.join(dirpath, 'frames/{}.jpg'.format(str(int(frameId*interval_frame)).zfill(6)))
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    wh  = np.array([[w, h]])
    plt.imshow(image)
    visual_data_pos = (1 + visual_data.x.numpy()[:, [1,0]]) * wh/2.0
    points_data_pos = (1 + points_data.x.numpy()[:, [1,0]]) * wh/2.0
    visual_data_label = visual_data.y.numpy()
    points_data_label = points_data.y.numpy()

    if with_edge:
        pass


    plt.scatter(points_data_pos[:, 0], points_data_pos[:, 1], marker='o', linewidths=2)
    plt.scatter(visual_data_pos[:, 0], visual_data_pos[:, 1], marker='*', linewidths=2)
    plt.show()


if __name__=='__main__':
    from torch.utils.data import DataLoader

    graph_data = graph_dataset(subsets=('zara01', 'zara02'))
    graph_dataloader = DataLoader(graph_data, batch_size=8, shuffle=True,
                                  collate_fn=graph_data.collate_fn, num_workers=1, pin_memory=True)
    for k, v in enumerate(graph_dataloader):
        visualize_graphs(v[0][0], v[1][0], v[2][0])
        break
