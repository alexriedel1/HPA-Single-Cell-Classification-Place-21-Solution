import os

class CFG:
    debug=False
    verbose = False
    num_workers=8
    model_name_effnet = 'efficientnet-b4' #'resnest101' #resnest50
    model_name_resnest = 'resnest101'
    size=640 #640
    seed=2002
    classes = 19
    color_mode = "rgby"
    resnest = True
    effnet = True
    extra_model_for_labels = True
    extra_model_is_tf = True
    only_green_extra_model = True
    color_mode_image_level = "rgb"
    split = [0.6, 0.4, 0] #mask_probas, img_level_model, mask_model
    size_seg = None
    split_image_level = [0.33, 0.33, 0.34, 0] #effnet ,resnest ,vit, densenet ::: image level
    split_cam_level = [0.5, 0.5] #effnet cam level, resnest cam level
    split_sigmoid_graboost = [0.5, 0.5]
    sigmoid_factor = 2.0
    sigmoid_move = 0.2
    is_demo = False
    batch_size_ = 4
    PATH_TEST = os.getcwd() + "/inference/test-images"
    
if CFG.is_demo:
    data_df = data_df[:10]

