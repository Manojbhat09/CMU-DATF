import os

TRAIN = True
 

def update_config_test(configs):
    # dataset specific 
    shuffle = False 
    pin_memory = False 
    
    # Dataset
    intrinsic_rate = 10
    max_distance = 56
    multi_agent = True
    use_scene = True
    scene_size = (60, 60)
    sample_stride = 1 
    sampling_rate = 2
    
    # Model 
    use_scene = False
    scene_size = (60, 60)
    model_name = "SocialPooling"
    TAG = "social_pooling"
    agent_embed_dim = 128
    nfuture = 12 # redundant if using sampling ratesa
    lstm_layers = 1
    lstm_dropout = 0.3
    pooling_size = 30
    
    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"
    test_ckpt = "../checkpoints/carla_MATF_D128.pth.tar"

    # Test
    batch_size = 1
    test_set = False 
    return configs

# Overrides
def update_config_train(configs):

    # dataset
    shuffle = True 
    pin_memory = True 
    device = "cuda"
    max_distance = 56
    multi_agent = True
    sample_stride = 1 
    sampling_rate = 2
     
    intrinsic_rate = 10

    # Model 
    use_scene = False
    scene_size = (60, 60)
    model_name = "SocialPooling"
    TAG = "social_pooling"
    agent_embed_dim = 128
    nfuture = 12 # redundant if using sampling ratesa
    lstm_layers = 1
    lstm_dropout = 0.3
    pooling_size = 30
    
    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"
    test_ckpt = "../checkpoints/SimpleEncDec.pth.tar"

    # Train     
    optimizer_name = "adam"
    batch_size = 4
    train = True
    validate = True

    return configs

if __name__=="__main__":
    print("Call update_config method")
