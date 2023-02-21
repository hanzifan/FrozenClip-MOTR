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
    args = parser.parse_args()

    device='cuda'
    model, preprocess = clip.load("RN50x64", device=device)
    model.float()
    main_path = os.path.join(args.base_path, args.data_name)

    path = os.path.join(main_path, 'images/track/train')
    sub_dir_list = glob.glob(path + "/*")
    label_list = clip.tokenize(["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]).to(device)
    image_prompt = {}
    text_prompt = {}
    bdd_prompt = {}

    with torch.no_grad():
        clip_img_emb = model.encode_text(label_list).float().cpu().detach()
        clip_img_emb = clip_img_emb.view(-1, clip_img_emb.shape[-2], clip_img_emb.shape[-1])
    text_prompt = clip_img_emb
    save_path = os.path.join('/home/hzf/project/MOTRv2/', 'clip-preprocessing')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'text-embedding.npy'), text_prompt)

    # for i in tqdm.tqdm(range(len(sub_dir_list))):
    #     sub_dir_name = sub_dir_list[i]
    #     imgae_list = glob.glob(sub_dir_name + "/*")
    #     for name in imgae_list:
    #         current_id = os.path.splitext(os.path.basename(name))[0]

    #         image = preprocess(Image.open(name)).unsqueeze(0).cuda()
    #         with torch.no_grad():
    #             prompt_embedding = model.encode_image(image).float().cpu().detach()

    #         image_prompt[current_id] = prompt_embedding

    #     if i == 200:
    #         save_path = os.path.join(main_path, 'clip-preprocessing')
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         np.save(os.path.join(save_path, 'image-embedding_clip.npy'), image_prompt)

    # save_path = os.path.join(main_path, 'clip-preprocessing')
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # np.save(os.path.join(save_path, 'image-embedding.npy'), image_prompt)
    # print('done!')

