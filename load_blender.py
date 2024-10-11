import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]
]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])) @ c2w
    return c2w


def load_blender_data(basedir, train_subset=None, test_subset=None, val_subset=None, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    subsets = {
        'train': train_subset,
        'val': val_subset,
        'test': test_subset
    }
    metas = {}
    for s in splits:
        subset = subsets[s]
        if subset is not None:
            # 不再拼接 's'，因为 'file_path' 已包含
            subset_dir = os.path.join(basedir, s, subset)
        else:
            subset_dir = os.path.join(basedir, s)

        transforms_path = os.path.join(subset_dir, f'transforms_{s}.json')

        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"Transform file not found: {transforms_path}")

        with open(transforms_path, 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        subset = subsets[s]
        if subset is not None:
            subset_dir = os.path.join(basedir, s, subset)
        else:
            subset_dir = os.path.join(basedir, s)

        for frame in meta['frames'][::skip]:
            file_path = frame['file_path']

            # 移除前缀 'train/', 'val/' 或 'test/'，假设 file_path 包含
            if s in ['train', 'val', 'test'] and file_path.startswith(s + '/'):
                file_path = file_path[len(s) + 1:]

            # 检查是否已经有 '.png' 扩展名
            if not file_path.endswith('.png'):
                fname = os.path.join(subset_dir, file_path + '.png')
            else:
                fname = os.path.join(subset_dir, file_path)

            if not os.path.exists(fname):
                raise FileNotFoundError(f"Image file not found: {fname}")

            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # 保留所有4个通道 (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0)
                                for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split
