import os 
from os import path as osp
import argparse

import numpy as np

import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from visualize import vis_utils as utils


class Rope3DReader(object):
    """
    fileio for read rope3d dataset
    """
    def __init__(self, root_path, split):
        self.root_path = root_path 
        assert split in ['training', 'validation'], 'Invalid split: {}'.format(split)
        self.split = split 
        
        # load labels in memory
        label_file = 'train.txt' if self.split == 'training' else 'val.txt'
        label_file = osp.join(self.root_path, self.split, label_file)
        self.labels = [line.strip().split(' ') for line in open(label_file, 'r').readlines()]

    def load_img(self, filename):
        filepath = osp.join(self.root_path, self.split + '-image_2', filename + '.jpg')
        return cv2.imread(filepath)
    
    def load_depth(self, filename):
        filepath = osp.join(self.root_path, self.split + '-depth_2', filename + '.jpg')
        return cv2.imread(filepath)
    
    def load_label(self, filename):
        filename = osp.join(self.root_path, self.split, 'label_2', filename + '.txt')
        return utils.read_label(filename)         
    
    def load_gplane(self, filename):
        filename = osp.join(self.root_path, self.split, 'denorm', filename + '.txt')
        handler = open(filename, 'r')
        lines = [line.strip().split(' ') for line in handler.readlines()]
        coef = list(map(lambda x: float(x), lines[0]))
        return np.array(coef, dtype=np.float32)
    
    def load_calib(self, filename):
        filename = osp.join(self.root_path, self.split, 'calib', filename + '.txt')
        return utils.Calibration(filename)
    
class RopeDataset(Dataset):
    def __init__(self, root_path, split='training'):
        super(RopeDataset, self).__init__()
        self.root_path = root_path
        self.split = split 
        self.data_reader = Rope3DReader(root_path, split)
        self.sample_set = self.load_set()

    def load_set(self):
        filename = 'train.txt' if self.split == 'training' else 'val.txt'
        filename = osp.join(self.root_path, self.split, filename)
        handler = open(filename, 'r')
        return [line.strip() for line in handler.readlines()]

    def __len__(self):
        return len(self.sample_set)
  
    def __getitem__(self, index):
        filename = self.sample_set[index]
        data = {
            'image': self.data_reader.load_img(filename),
            'calib': self.data_reader.load_calib(filename),
            'gplane': self.data_reader.load_gplane(filename),
            'labels': self.data_reader.load_label(filename),
            'name': filename,
            'idx': index
        }
        return data 

def show_image_with_boxes(img, objects, calib, gplane, name='0', vis_2d=False):
    """ Show image with 2D/3D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    for idx, obj in enumerate(objects):
        cv2.rectangle(
        img1,
        (int(obj.xmin), int(obj.ymin)),
        (int(obj.xmax), int(obj.ymax)),
        (0, 255, 0), thickness=2)

        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P, gplane)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))

    if not os.path.exists('./vis_imgs/'):
        os.mkdir('./vis_imgs')
    if vis_2d:
        cv2.imwrite('./vis_imgs/{}_2d.jpg'.format(name), img1)
    cv2.imwrite('./vis_imgs/{}_3d.jpg'.format(name), img2)

def get_args_parser():
    parser = argparse.ArgumentParser('VLC Equalizer', add_help=False)
    parser.add_argument('--num_img', default=100, type=int)
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--split', default='training', type=str)
    parser.add_argument('--vis_2d', dest='vis_2d', action='store_true')
    parser.add_argument('--show_name', dest='show_name', action='store_true')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Rope3D Dataset.', parents=[get_args_parser()])
    args = parser.parse_args()
    dataset = RopeDataset(args.data_root, split=args.split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    for idx, data in enumerate(dataloader):
        if idx == args.num_img:
            break 
        data = data[0]
        name = data['name'] if args.show_name else data['idx']
        show_image_with_boxes(data['image'], data['labels'], 
                                data['calib'], data['gplane'], 
                                name=name, vis_2d=args.vis_2d)
