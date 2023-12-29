import os
import sys

import json
import yaml
import pickle
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))
from third_parties.smpl.smpl_numpy import SMPL

from absl import flags
FLAGS = flags.FLAGS
import argparse
import os
MODEL_DIR = '../../third_parties/smpl/models'



def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def main(argv):

    dataset_path = argv.dataset_path

    sex = 'neutral'

    output_path = dataset_path
    
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        frame_infos = json.load(f)

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []

    for frame_base_name in tqdm(frame_infos):

        cam_body_info = frame_infos[frame_base_name] 

        poses = np.array(cam_body_info['poses'], dtype=np.float32)[0]
        betas = np.array(cam_body_info['betas'], dtype=np.float32)[0]
        K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
        E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)
        
        all_betas.append(betas)

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        _, _, tpose_joints = smpl_model(np.zeros_like(poses), betas)

        # get global Rh, Th
        pelvis_pos = tpose_joints[0].copy()
        Th = pelvis_pos
        Rh = poses[:3].copy()

        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # remove global rotation from body pose
        poses[:3] = 0

        # get posed joints using body poses without global rotation
        _, _, joints = smpl_model(poses, betas)
        joints = joints - pelvis_pos[None, :]

        mesh_infos[frame_base_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'beats': betas,
            'joints': joints, 
            'tpose_joints': tpose_joints
        }


        cameras[frame_base_name] = {
            'intrinsics': K,
            'extrinsics': E
        }


    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)


def parse_args():
    """
    Args:

    """
    parse = argparse.ArgumentParser(description='Prepare wild data')
    parse.add_argument('--dataset_path', type=str, required=True, help='wild dataset path')
    args = parse.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)



