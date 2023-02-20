import torch
import os
import torch.nn as nn
import numpy as np

class Distillation_Correlation(nn.Module):
    def __init__(self, T=4, prompt_type='img', scale=2):
        super(Distillation_Correlation, self).__init__()
        # preprocessed clip embedding path
        self.train_clip_preprocessing_path = "/home/cc/dataset/super_resolution/COCO/clip-preprocessing"
        self.test_clip_preprocessing_path = "/home/cc/dataset/super_resolution/Set5/clip-preprocessing"
        # NOTE: distillation part
        # sr feature projection layer for distillation
        self.proj_channel = nn.Conv2d(64, 1024, kernel_size=1, bias=False)
        torch.nn.init.uniform_(self.proj_channel.weight, a=-0.1, b=0.1)
        # define distillation loss function
        self.loss = nn.L1Loss()
        # preprocessed LR image clip embedding
        self.dis_prompt = np.load(
            os.path.join(self.train_clip_preprocessing_path, 'lr-x{}-embedding.npy'.format(scale)),
            allow_pickle=True).item()
        self.dis_prompt_inference = np.load(
            os.path.join(self.test_clip_preprocessing_path, 'lr-x{}-embedding.npy'.format(scale)),
            allow_pickle=True).item()

        # NOTE: correlation and dynamic part
        # T for computing scaled pairwise cosine similarities
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)
        # dynamic convolution layer for reducing channels
        self.post_conv = Dynamic_conv2d(1024, 64, kernel_size=1, bias=False, temperature=T)
        torch.nn.init.uniform_(self.post_conv.weight, a=-0.01, b=0.01)
        # residual scale
        self.ratio = 0.1

        # preprocessed HR image and caption clip embedding
        self.image_prompt = np.load(
            os.path.join(self.train_clip_preprocessing_path, 'hr-embedding.npy'),
            allow_pickle=True).item()
        self.text_prompt = np.load(
            os.path.join(self.train_clip_preprocessing_path, 'text-embedding.npy'),
            allow_pickle=True).item()
        self.image_prompt_inference = np.load(
            os.path.join(self.test_clip_preproce