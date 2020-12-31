import os

TRAIN = True
 

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

    # Model 
    model_name = "MATF"
    TAG = "matf"
    device = "cuda"
    agent_embed_dim = 128
    nfuture = 12 # redundant if using sampling ratesa
    lstm_layers = 1
    lstm_dropout = 0.3
    encoder_type="ShallowCNN" # "ShallowCNN | ResNet"
    scene_channels= 3
    scene_dropout=0.5
    freeze_resnet=True
    disc_hidden = 512
    disc_dropout = 0.5
    gan_weight_schedule = [20, 30, 40, 50, 65, 200]
    gan_weight = [0.5, 0.7, 1, 1.5, 2.0, 2.5]
    
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
    device = "cuda"
     
    intrinsic_rate = 10
    max_distance = 56
    multi_agent = True
    use_scene = True
    scene_size = (60, 60)
    sample_stride = 1 
    sampling_rate = 2
    shuffle = False 
    pin_memory = False 

    # Model
    model_name = "MATF_GAN"
    TAG = "matf_gan"
    agent_embed_dim = 128
    nfuture = 12 # redundant if using sampling ratesa
    lstm_layers = 1
    lstm_dropout = 0.3
    encoder_type="ShallowCNN" # "ShallowCNN | ResNet"
    freeze_resnet=True
    scene_channels= 1
    scene_dropout=0.5
    disc_hidden = 512
    disc_dropout = 0.5
    gan_weight_schedule = [20, 30, 40, 50, 65, 200]
    gan_weight = [0.5, 0.7, 1, 1.5, 2.0, 2.5]
    generative = True
    map_version = '2.0'
    scene_channels = 5 if map_version == '2.1' else 3

    # Paths
    root_dir = "/data/datasets/datf/CMU-DATF"
    cache_path = "/data/datasets/datf/CMU-DATF/caches"
    exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    optimizer_name = ["adam", "adam"] # Generator and Discriminator
    batch_size = 2
    train = True
    validate = True
    return configs

if __name__=="__main__":
    print("Call update_config method")
