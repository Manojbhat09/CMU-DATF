import os

TRAIN = True
  

def update_config_train(configs):

    # Dataset
    device = "cuda"
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
    model_name = "Global_Scene_CAM_NFDecoder"
    TAG = "global_scene_nfdecode"
    agent_embed_dim = 128
    scene_channels = 5 if map_version == '2.1' else 3
    nfuture = int(3 * sampling_rate)

    # nfuture = 12 # redundant if using sampling ratesa
    att_dropout = 0.1
    lstm_layers = 1
    lstm_dropout = 0.3
    velocity_const = 0.5
    num_candidates = 12
    decoding_steps = nfuture
    att = False
    ploss_type = "map"
    ploss_criterion = "Interpolated_Ploss"

    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    optimizer_name = "adam"
    batch_size = 4
    train = True
    validate = True
    return configs

def update_config_test(configs):

    # Dataset
    device = "cuda"
     
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
    model_name = "Global_Scene_CAM_NFDecoder"
    TAG = "global_scene_nfdecode"
    agent_embed_dim = 128
    nfuture = 12 # redundant if using sampling ratesa
    att_dropout = 0.1
    lstm_layers = 1
    lstm_dropout = 0.3
    velocity_const = 0.5
    num_candidates = 12
    decoding_steps = 6
    att = False
    ploss_type = "map"
    ploss_criterion = "Interpolated_Ploss" 

    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    optimizer_name = "adam"
    batch_size = 4
    train = True
    validate = True
    return configs


if __name__=="__main__":
    print("Call update_config method")
