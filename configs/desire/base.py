import os

TRAIN = True
 

def update_config_train(configs):

    # Dataset
    device = "cuda" 
    intrinsic_rate = 10
    max_distance = 56
    multi_agent = True
    use_scene = True
    scene_size = [64, 64]
    sample_stride = 1 
    sampling_rate = 2
    shuffle = False 
    pin_memory = False 

    # Model
    model_name = "DESIRE"
    TAG = "desire"
    ploss_type = "map"
    # ploss = "Interpolated_Ploss"
    num_candidates = 12
    map_version = '2.0'
    nfuture = int(3 * sampling_rate)
    decoding_steps = nfuture
    in_channels = 5 if map_version == '2.1' else 3

    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    optimizer_name = ["adam","adam"]
    batch_size = 4
    train = True
    validate = True
    return configs


def update_config_test(configs):

    # Dataset
    intrinsic_rate = 10
    max_distance = 56
    multi_agent = True
    use_scene = True
    scene_size = (60, 60)
    sample_stride = 1 
    sampling_rate = 2
    shuffle = False 
    pin_memory = False 
    device = "cuda"

    # Model
    model_name = "DESIRE"
    TAG = "desire"
    ploss_type = "mseloss"
    num_candidates = 12
    map_version = '2.1'
    decoding_steps = 6
    in_channels = 5 if map_version == '2.1' else 3
    
    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"
    test_ckpt = "../checkpoints/carla_MATF_D128.pth.tar"

    # Test
    batch_size = 1
    test_set = False 
    return configs

if __name__=="__main__":
    print("Call update_config method")
