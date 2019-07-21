import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os.path as osp
import numpy as np
import os
os.chdir('/data/zwzhou/TestCode/Hope')
from dataset.graph_simulation import simulate_visual, simulate_points, RandomTranslate

import cv2
import matplotlib.pyplot as plt



def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary[0])
    return result

class graph_dataset(Dataset):
    def __init__(self, subsets=('zara01, hotel, univ'), loss_prob=0.2, add_prob=0.3):
        super(graph_dataset, self).__init__()
        self.loss_prob = loss_prob
        self.add_prob = add_prob
        records = []
        for subset in subsets:
            if subset == 'zara01':
                dirpath = 'data/UCY/zara/zara01'
            elif subset == 'zara02':
                dirpath = 'data/UCY/zara/zara02'
            elif subset == 'univ':
                dirpath = 'data/UCY/univ'
            elif subset == 'eth':
                dirpath = 'data/ETH/seq_eth'
            elif subset == 'hotel':
                dirpath = 'data/ETH/seq_hotel'
            else:
                raise Exception('No dataset named: {}'.format(subset))
            filepath = osp.join(dirpath, 'records.npy')
            records.append(np.load(filepath, allow_pickle=True)[()])  # np saved dict
        self.records = merge_dicts(records)
        self.key_index = {i: key for i, key in enumerate(self.records.keys())}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        key = self.key_index[item]
        record = self.records[key]
        x, y = np.array(record.x), np.array(record.y)

        # for visualize
        visual_x, visual_y = RandomTranslate(*simulate_visual(x, y, self.loss_prob))
        grid_x, grid_y = np.meshgrid(np.arange(len(visual_x)), np.arange(len(visual_y)))
        mask = np.eye(len(visual_x)).flatten()
        visual_edge_index = np.stack((grid_x.flatten(), grid_y.flatten()), axis=0)[:, mask==0]
        visual_x = torch.from_numpy(visual_x)
        visual_y = torch.from_numpy(visual_y)
        visual_edge_index = torch.from_numpy(visual_edge_index).long()
        visual_graph = Data(x=visual_x, edge_index=visual_edge_index, y=visual_y)
        # for points
        points_x, points_y = RandomTranslate(*simulate_points(x, y, self.add_prob))
        grid_x, grid_y = np.meshgrid(np.arange(len(points_x)), np.arange(len(points_y)))
        mask = np.eye(len(points_x)).flatten()
        points_edge_index = np.stack((grid_x.flatten(), grid_y.flatten()), axis=0)[:, mask==0]
        points_x = torch.from_numpy(points_x)
        points_y = torch.from_numpy(points_y)
        points_edge_index = torch.from_numpy(points_edge_index).long()
        points_graph = Data(x=points_x, y=points_y, edge_index=points_edge_index)
        return visual_graph, points_graph, key

    @staticmethod
    def collate_fn(batch):
        visual_graph_list, points_graph_list, key_list = zip(*batch)
        return visual_graph_list, points_graph_list, key_list


if __name__=='__main__':
    from torch.utils.data import DataLoader
    graph_data = graph_dataset(subsets=('zara01','zara02'))
    graph_dataloader = DataLoader(graph_data, batch_size=8, shuffle=True,
                                  collate_fn=graph_data.collate_fn, num_workers=1, pin_memory=True)
    for k,  v in enumerate(graph_dataloader):
        print(len(v[0]))
        print(v[0][0].x)
        print(v[0][0].num_nodes)
        print(v[0][0].num_edges)
        print(v[0][0].num_node_features)
        print(v[1][0].x)
        print(v[1][0].num_nodes)
        print(v[1][0].num_edges)
        print(v[1][0].num_node_features)
        print(v[2][0])
        break
