import os

TRAIN = True
 

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
    val_check = True
    train = True

    # Model 
    use_scene = False
    scene_size = (60, 60)
    model_name = "SimpleEncoderDecoder"
    TAG = "SimpleModel_rpn1_dir"
    agent_embed_dim = 128
    nfuture = 12 # redundant if using sampling ratesa
    lstm_layers = 1
    lstm_dropout = 0.3
    
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

# Overrides
def update_config_test(configs):

    # dataset
    shuffle = True 
    pin_memory = True 
    device = "cuda"
    max_distance = 56
    multi_agent = True
    sample_stride = 1 
    sampling_rate = 2
     
    intrinsic_rate = 10
    val_check = True
    train = False

    # Model 
    use_scene = False
    scene_size = (60, 60)
    model_name = "SimpleEncoderDecoder"
    TAG = "SimpleModel_rpn1_dir"
    agent_embed_dim = 128
    nfuture = 12 # redundant if using sampling ratesa
    lstm_layers = 1
    lstm_dropout = 0.3
    
    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"
    test_ckpt = "../checkpoints/SimpleEncDec.pth.tar"

    # Train     
    optimizer_name = "adam"
    batch_size = 4
    test_set = True
    return configs

if __name__=="__main__":
    print("Call update_config method")
