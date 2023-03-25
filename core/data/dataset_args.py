from configs import cfg, args

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394']

    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap':
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "index_a": cfg.index_a,
                    "index_b": cfg.index_b,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "index_a": cfg.index_a,
                    "index_b": cfg.index_b,
                },
            })

    subjects = ['wild']
    if cfg.category == 'human_nerf' and cfg.task == 'wild':
        for sub in subjects:
            dataset_attrs.update({
                f"monocular_{sub}_train": {
                    "dataset_path":f"dataset/wild/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "start": cfg.start,
                    "end": cfg.end,
                    "index_a": cfg.index_a,
                    "index_b": cfg.index_b,
                },
                f"monocular_{sub}_test": {
                    "dataset_path":f"dataset/wild/{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'wild',
                    "start": cfg.start,
                    "end": cfg.end,
                    "index_a": cfg.index_a,
                    "index_b": cfg.index_b,
                },
            })


    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
