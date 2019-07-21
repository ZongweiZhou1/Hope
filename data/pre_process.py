import numpy as np
import json
from easydict import EasyDict as edict
import os.path as osp

import cv2
import matplotlib.pyplot as plt


DEBUG = False

def read_pos_file(dataset):
    """ re-organize data
    :param dataset:         in choices [zara01, zara02, univ, eth, hotel]
    :return:
    """
    if dataset == 'zara01':
        dirpath = 'UCY/zara/zara01/'
    elif dataset == 'zara02':
        dirpath = 'UCY/zara/zara02/'
    elif dataset == 'univ':
        dirpath = 'UCY/univ'
    elif dataset == 'eth':
        dirpath = 'ETH/seq_eth'
    elif dataset == 'hotel':
        dirpath = 'ETH/seq_hotel'
    else:
        raise Exception('No dataset: `{}` found!'.format(dataset))

    filepath = osp.join(dirpath, 'pixel_pos_interpolate.csv')
    records = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        anno_data = np.array([list(map(float, items.split(','))) for items in lines]).astype(np.float32).T
        anno_data = anno_data[np.argsort(anno_data[:, 0])]
        # each row in anno_data corresponding to (frameId, trackid, pos_x, pos_y)
        for frame in set(anno_data[:, 0]):  # iterate all frames with annotations
            prev_frame_data = anno_data[anno_data[:, 0]==frame-1]
            curr_frame_data = anno_data[anno_data[:, 0]==frame]
            next_frame_data = anno_data[anno_data[:, 0]==frame+1]
            x, y = [], []
            for trackid in curr_frame_data[:, 1]:
                v = np.array([[0.0, 0.0]]).astype(np.float32)
                n = 0
                pos = curr_frame_data[curr_frame_data[:, 1]==trackid, 2:]
                if trackid in prev_frame_data[:, 1]:
                    v += pos - prev_frame_data[prev_frame_data[:, 1]==trackid, 2:]
                    n += 1
                if trackid in next_frame_data[:, 1]:
                    v += next_frame_data[next_frame_data[:, 1]==trackid, 2:] - pos
                    n += 1
                if n==0: n = 1
                x.append([pos[0, 0], pos[0, 1], v[0, 0]/n, v[0, 1]/n])
                y.append(trackid)
            records[(dataset, frame)] = edict()
            records[(dataset, frame)].x = x
            records[(dataset, frame)].y = y
            if DEBUG:
                img = cv2.imread(osp.join(dirpath, 'frames/{}.jpg'.format(str(int(frame*8)).zfill(6))))
                h, w = img.shape[:2]
                hw = np.array([[h, w]])
                x = np.array(x)
                x = (1 + x[:, :2]) * hw/2.0

                plt.imshow(img)
                plt.scatter(x[:, 1], x[:, 0])
                plt.show()

    np.save(osp.join(dirpath, 'records.npy'), records)
    print('The data from {} have been saved to {}.'.format(filepath, osp.join(dirpath, 'records.npy')))
    return records

if __name__=='__main__':
    for dataset in ['zara01', 'zara02', 'univ', 'eth', 'hotel']:
        read_pos_file(dataset)



