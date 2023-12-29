from tqdm import tqdm
import joblib
import os
import json
import numpy as np
import argparse

def convert_weak_perspective_to_perspective(
        weak_perspective_camera,
        focal_length=5000.,
        img_res=224,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    perspective_camera = np.stack(
        [
            weak_perspective_camera[1],
            weak_perspective_camera[2],
            2 * focal_length / (img_res * weak_perspective_camera[0] + 1e-9)
        ],
        axis=-1
    )
    return perspective_camera

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description='Prepare wild data')
    parse.add_argument('--dataset_path', type=str, required=True, help='wild dataset path')
    args = parse.parse_args()
    datadir = args.dataset_path

    filenames = sorted(os.listdir(os.path.join(datadir, 'images')))
    pare_focal, pare_h = 5000.0, 224
    orig_h = 224
    metadata = {}
    
    for filename in tqdm(filenames):
        pare_result = joblib.load(os.path.join(datadir, 'pare', filename[:-4]+'.pkl'))
        print(pare_result.keys())
        pose = np.array(pare_result['pose_vec']).reshape(-1,72)
        beta = np.array(pare_result['pred_shape'])
        orig_cam = np.array(pare_result['orig_cam'])
        pred_cam = np.array(pare_result['pred_cam'])[0]
        bbox = np.array(pare_result['bboxes'])[0]

        RT = np.eye(4)
        RT[:3, 3] = convert_weak_perspective_to_perspective(pred_cam)
        cx, cy, h = bbox[0], bbox[1], bbox[2]
        f = h / pare_h * pare_focal
        K = np.eye(3)
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
        meta = {
            "poses": pose.tolist(),
            "betas": beta.tolist(),
            "cam_intrinsics": K.tolist(),
            "cam_extrinsics": RT.tolist(),
        }
        metadata[filename[:-4]] = meta

    with open(os.path.join(datadir, 'metadata.json'), 'w') as jsonfile:
        json.dump(metadata, jsonfile, indent=4)

