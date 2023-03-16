import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import imageio
import torchvision
from network import RAFTGMA, RAFTGMA_QuadtreeFnet
import torch.nn.functional as F
import datasets
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission_vis(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            original_shape = image1.shape[2:]
            new_shape = int(np.ceil(original_shape[0] / 32) * 32), int(np.ceil(original_shape[1] / 32) * 32)
            image1 = F.interpolate(image1, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)

            flow_low, flow_pr = model(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)

            flow = F.interpolate(flow_pr, (original_shape[0], original_shape[1]), mode='bilinear', align_corners=True)[
                       0] \
                       .cpu() * torch.tensor([original_shape[1] / new_shape[1], original_shape[0] / new_shape[0]]).view(
                2, 1, 1)
            flow = flow.permute(1, 2, 0).numpy()

            # Visualizations
            flow_img = flow_viz.flow_to_image(flow)
            image = Image.fromarray(flow_img)

            if not os.path.exists(f'vis_test/sintel_submission/{dstype}/flow/{sequence}'):
                os.makedirs(f'vis_test/sintel_submission/{dstype}/flow/{sequence}')

            if not os.path.exists(f'vis_test/gt/{dstype}/image/{sequence}'):
                os.makedirs(f'vis_test/gt/{dstype}/image/{sequence}')

            image.save(f'vis_test/sintel_submission/{dstype}/flow/{sequence}/{test_id}.png')
            imageio.imwrite(f'vis_test/gt/{dstype}/image/{sequence}/{test_id}.png',
                            image1[0].cpu().permute(1, 2, 0).numpy())

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=24):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation')
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            original_shape = image1.shape[2:]
            new_shape = int(np.ceil(original_shape[0] / 32) * 32), int(np.ceil(original_shape[1] / 32) * 32)
            image1 = F.interpolate(image1, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)

            flow = F.interpolate(flow_pr, (original_shape[0], original_shape[1]), mode='bilinear', align_corners=True)[
                       0] \
                       .cpu() * torch.tensor([original_shape[1] / new_shape[1], original_shape[0] / new_shape[0]]).view(2, 1, 1)

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()

            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_occ(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []
        epe_in_frame = []
        epe_out_frame = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]

            # in and out of frame occu
            _, h, w = image1.size()
            coords = torch.meshgrid(torch.arange(h), torch.arange(w))
            coords = torch.stack(coords[::-1], dim=0).float()

            coords_img_2 = coords + flow_gt
            out_of_frame = (coords_img_2[0] < 0) | (coords_img_2[0] > w) | (coords_img_2[1] < 0) | (coords_img_2[1] > h)
            occ_union = out_of_frame | occ
            in_frame = occ_union ^ out_of_frame

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            original_shape = image1.shape[2:]
            new_shape = int(np.ceil(original_shape[0] / 32) * 32), int(np.ceil(original_shape[1] / 32) * 32)
            image1 = F.interpolate(image1, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = F.interpolate(flow_pr, (original_shape[0], original_shape[1]), mode='bilinear', align_corners=True)[
                       0] \
                       .cpu() * torch.tensor([original_shape[1] / new_shape[1], original_shape[0] / new_shape[0]]).view(
                2, 1, 1)

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())
            epe_in_frame.append(epe[in_frame].numpy())
            epe_out_frame.append(epe[out_of_frame].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)
        epe_in_frame = np.concatenate(epe_in_frame)
        epe_out_frame = np.concatenate(epe_out_frame)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)
        epe_in_frame_mean = np.mean(epe_in_frame)
        epe_out_frame_mean = np.mean(epe_out_frame)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f, Occ-in: %f, Occ-out: %f" % (epe_occ_mean, epe_noc_mean, epe_in_frame_mean, epe_out_frame_mean))
        results[dstype] = np.mean(epe_list)

    return results
