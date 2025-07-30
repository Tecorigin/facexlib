import argparse
import cv2
import glob
import math
import numpy as np
import os
import torch

from facexlib.recognition import init_recognition_model


def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = image.astype(np.float32, copy=False)
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
    image -= 127.5
    image /= 127.5
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)  # Change to CxHxW
    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type=str, default='assets/folder1')
    parser.add_argument('--folder2', type=str, default='assets/folder2')
    parser.add_argument('--model_name', type=str, default='arcface')

    args = parser.parse_args()

    img_list1 = sorted(glob.glob(os.path.join(args.folder1, '*')))
    img_list2 = sorted(glob.glob(os.path.join(args.folder2, '*')))
    print(img_list1, img_list2)
    model = init_recognition_model(args.model_name)

    dist_list = []
    identical_count = 0
    for idx, (img_path1, img_path2) in enumerate(zip(img_list1, img_list2)):
        basename = os.path.splitext(os.path.basename(img_path1))[0]
        img1 = load_image(img_path1)
        img2 = load_image(img_path2)

        data = torch.stack([img1, img2], dim=0)
        data = data.to(torch.device('sdaa'))
        output = model(data)
        print(output.size())
        output = output.data.cpu().numpy()
        dist = cosin_metric(output[0], output[1])
        dist = np.arccos(dist) / math.pi * 180
        print(f'{idx} - {dist} o : {basename}')
        if dist < 1:
            print(f'{basename} is almost identical to original.')
            identical_count += 1
        else:
            dist_list.append(dist)

    print(f'Result dist: {sum(dist_list) / len(dist_list):.6f}')
    print(f'identical count: {identical_count}')
