import os

TRAIN = True
 

def update_config_test(configs):
        # Dataset
    intrinsic_rate = 10
    max_distance = 56
    multi_agent = True
    use_scene = True
    scene_size = (64, 64)
    sample_stride = 1 
    sampling_rate = 2
    shuffle = False 
    pin_memory = False 
    device = "cuda"

    # Model 
    model_name = "R2P2_RNN"
    TAG = "r2p2_rnn"
    map_version = '2.1'
    scene_channels = 5 if map_version == '2.1' else 3
    velocity_const = 0.5 
    num_candidates = 12
    decoding_steps = 6
    nfuture = 12
    
    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"
    test_ckpt = "../checkpoints/carla_MATF_D128.pth.tar"

    # Test
    batch_size = 1
    test_set = False 
    return configs

def update_config_train(configs):

    # Dataset
    dataset="argoverse_r2p2"
    device = "cuda"
    ploss_criterion = "MSE" 
    intrinsic_rate = 10
    max_distance = 56
    multi_agent = True
    use_scene = True
    scene_size = (64, 64)
    sample_stride = 1 
    sampling_rate = 2
    shuffle = False 
    pin_memory = False 

    # Model
    # model_name = "R2P2_RNN"
    model_name = "R2P2_SimpleRNN"
    TAG = "r2p2_rnn_simple"
    map_version = '2.0'
    scene_channels = 5 if map_version == '2.1' else 1
    velocity_const = 0.5 
    num_candidates = 12
    decoding_steps = 6
    ploss_type = 'mseloss' # 'interpolated_ploss'
    nfuture = 12
    generative=False
    # beta = 0.0
    beta = 0.1

    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    optimizer_name = "adam"
    batch_size = 4
    train = True
    validate = True
    num_epochs = 1
    return configs

if __name__=="__main__":
    print("Call update_config method")
