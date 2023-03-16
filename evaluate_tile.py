import sys

sys.path.append('core')

from PIL import Image
import os
import numpy as np
import torch
import torch.nn.functional as F
import datasets
from utils import flow_viz
from utils import frame_utils
import imageio
import math


def compute_grid_indices(image_shape, patch_size, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    # unique
    hs = np.unique(hs)
    # ws.append(32)
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0], w:w + patch_size[1]])

    return patch_weights


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission', sigma=0.05):
    """ Create submission for the KITTI leaderboard """

    IMAGE_SIZE = [384, 1242]
    TRAIN_SIZE = [288, 960]

    print(f"output path: {output_path}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"training size: {TRAIN_SIZE}")

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:  # fix the height=432, adaptive ajust the width
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        image1, image2 = image1[None].cuda(), image2[None].cuda()
        image1 = F.interpolate(image1, (384, new_shape[1]), mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, (384, new_shape[1]), mode='bilinear', align_corners=True)

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
            _, flow_pre = model.module(image1_tile, image2_tile, iters=24, test_mode=True)

            padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = F.interpolate(flow_pre, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)[0].cpu() * \
               torch.tensor([1, new_shape[0] / 384]).view(2, 1, 1)
        flow = flow.permute(1, 2, 0).numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')
            os.makedirs(f'vis_kitti/image')

        image.save(f'vis_kitti/flow/{test_id}.png')
        imageio.imwrite(f'vis_kitti/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        imageio.imwrite(f'vis_kitti/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())


@torch.no_grad()
def validate_kitti(model, iters=24, sigma=0.05):
    IMAGE_SIZE = [416, 1242]
    TRAIN_SIZE = [416, 736]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        image1, image2 = image1[None].cuda(), image2[None].cuda()
        image1 = F.interpolate(image1, (416, new_shape[1]), mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, (416, new_shape[1]), mode='bilinear', align_corners=True)

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]

            _, flow_pre = model(image1_tile, image2_tile, iters=iters, test_mode=True)

            padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count

        flow = F.interpolate(flow_pre, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)[0].cpu() * \
               torch.tensor([1, new_shape[0] / 416]).view(2, 1, 1)

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

