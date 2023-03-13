import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import clip
from pycocotools.coco import COCO
import glob
import PIL.Image as Image
import tqdm
import torch
from einops import rearrange
import torch.nn.functional as F


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='clip preprocessing')
    parser.add_argument('--data_name', default='bdd100k')
    parser.add_argument('--base_path', default='/home/hzf/data/bdd/')
    parser.add_argument('--lvis_path', default='/data/hzf_data/coco/coco_labels.txt')
    args = parser.parse_args()

    # load clip model
    device='cuda'
    model, preprocess = clip.load("RN50", device=device)
    model.float()
    main_path = os.path.join(args.base_path, args.data_name)

    path = os.path.join(main_path, 'images/track/train')
    sub_dir_list = glob.glob(path + "/*")

    # generate lvis label
    lvis_label_list_path = open(args.lvis_path)
    lvis_label_list = lvis_label_list_path.read()
    lvis_label_list = lvis_label_list.split('\n')
    bdd_list = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    for idx in range(len(lvis_label_list)):
        if lvis_label_list[idx] in bdd_list:
            lvis_label_list[idx] = 'None'
        else:
            lvis_label_list[idx] = "This is a " + lvis_label_list[idx]
    print(len(lvis_label_list))
    label_emb = clip.tokenize(lvis_label_list).to(device)
    # image_prompt = {}
    text_prompt = {}

    # inference clip text encoder and save
    with torch.no_grad():
        clip_img_emb = model.encode_text(label_emb).float().cpu().detach()
        clip_img_emb = clip_img_emb.view(-1, clip_img_emb.shape[-2], clip_img_emb.shape[-1])
    text_prompt = clip_img_emb
    save_path = os.path.join('/home/hzf/project/MOTRv2/', 'clip-preprocessing')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'text-embedding_lvis.npy'), text_prompt)
