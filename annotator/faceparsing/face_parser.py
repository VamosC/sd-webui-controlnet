#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .model import BiSeNet

import torch
import torch.nn as nn

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from annotator.annotator_path import models_path


class FaceParser():
    """
    A class for face parsing

    Attributes:
        model_dir (str): Path to the directory where the face parser models are stored.
    """

    model_dir = os.path.join(models_path, "face-parsing")

    def __init__(self, n_classes=19):

        self.net = None
        self.n_classes = n_classes
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load_model(self):

        if self.net is None:
            net = BiSeNet(n_classes=self.n_classes)
            net = net.cuda()
            net.load_state_dict(torch.load(os.path.join(self.model_dir, '79999_iter.pth')))
            net.eval()
            self.net = net

    def unload_model(self):
        if self.net is not None:
            self.net = self.net.to('cpu')

    def vis_parsing_maps(self, im, parsing_anno, stride):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_parsing_anno_color = Image.fromarray(cv2.cvtColor(vis_parsing_anno_color, cv2.COLOR_RGB2BGR))

        return vis_parsing_anno_color

    @torch.no_grad()
    def __call__(self, image):

        if self.net is None:
            self.load_model()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img = self.transform(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = self.net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        vis_parsing = self.vis_parsing_maps(image, parsing, stride=1)

        image_wobg = np.asarray(image).copy()
        index = np.where((parsing == 0) | (parsing == 7) | (parsing == 8) | (parsing == 9) | (parsing == 14) | (parsing == 15) | (parsing == 16) | (parsing == 17) | (parsing == 18))
        image_wobg[index[0], index[1], :] = [255, 255, 255]
        image_wobg = Image.fromarray(image_wobg)

        return image_wobg, vis_parsing
