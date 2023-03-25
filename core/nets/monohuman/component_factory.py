import imp

def load_projector(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).Projector

def load_blend_net(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).BlendNet


def load_projection(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).ProjectionLayer

def load_positional_embedder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).get_embedder

def load_canonical_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).CanonicalMLP


def load_mweight_vol_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).MotionWeightVolumeDecoder


def load_pose_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).BodyPoseRefiner

def load_non_rigid_motion_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).NonRigidMotionMLP

def load_feature_extractor(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).ResUNet