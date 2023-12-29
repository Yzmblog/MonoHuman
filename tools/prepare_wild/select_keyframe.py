import pickle
import os
import numpy as np
import cv2
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--angle_threahold', type=int, default=30, required=False)     
    parser.add_argument('--dataset_path', type=str, required=True) 

    args = parser.parse_args()
    angle_threahold = args.angle_threahold
    dataset_path = args.dataset_path


    with open(os.path.join(dataset_path, 'mesh_infos.pkl'), 'rb') as f:
        mesh_infos = pickle.load(f)
    poses_a = []
    poses_b = []
    frames_a = []
    frames_b = []
    for frame_name in mesh_infos.keys():
        Rh = mesh_infos[frame_name]['Rh']
        pose = mesh_infos[frame_name]['poses'].astype('float32')
        Rot = cv2.Rodrigues(Rh)[0]
        idx = int(frame_name.split('_')[1])
        dir_vector = [0, 0, 1]
        dir_vector = np.array(dir_vector)
        dir_vector = dir_vector.dot(Rot.T)
        dir_vector[1] = 0

        z_axis = np.array([0, 0, 1])

        norm_vector1 = np.linalg.norm(dir_vector)

        cosine_angle = np.dot(dir_vector, z_axis) / (norm_vector1)
        # Calculate the angle in radians
        angle_radians = np.arccos(cosine_angle)
        # Convert the angle to degrees
        angle_degrees = np.degrees(angle_radians)

        front = angle_degrees >= 180 - angle_threahold
        back = angle_degrees <= angle_threahold

        if front:
            poses_a.append(pose)
            frames_a.append(frame_name)
        elif back:
            poses_b.append(pose)
            frames_b.append(frame_name)

    poses_a = np.array(poses_a)
    poses_b = np.array(poses_b)
    dis_metric = []

    for front_pose in poses_a:
        front_poses = np.repeat([front_pose], [poses_b.shape[0]], axis=0)
        l2 = np.linalg.norm(front_poses - poses_b, axis=1)
        dis_metric.append(l2)
    dis_metric = np.array(dis_metric)
    idx = dis_metric.argmin()
    index_a = idx // poses_b.shape[0]
    index_b = idx % poses_b.shape[0]
    index_a = frames_a[index_a]
    index_b = frames_b[index_b]

    print(f'Selecting results: index_a: {index_a}; index_b: {index_b}')
