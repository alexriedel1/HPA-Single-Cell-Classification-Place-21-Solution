class CFG:
    debug=False
    num_workers=8
    model_name= 'resnest101' #resnest101 #'efficientnet-b3
    size=640
    batch_size=8 #resnest101: 8 efficientnet-b4:6
    seed=2002
    T_0=8
    epochs = 10
    gradient_accumulation_steps=3  #3
    print_freq = 100
    lr= 2.5e-3  #was 5e-4 for resnest101, was 5e-4 for resnest50  #1.5e-3 for efficientnet b3 #1e-2 was good but too high in later, so use with better scheduling...
    min_lr = 2e-5
    weight_decay = 1e-4
    classes = 19
    num_pieces = 4 #for tiling!
    nesterov = True
    loss_option = "cl_pcl_re"
    alpha = 4.00
    alpha_schedule = 0.50
    level = "feature"
    re_loss_option = "masking"
    color_mode = "rgby"
    focal_gamma = 1.0
    resume = False
    MODEL_PATH = "models/" #path to model to resume from
    focal_type = 1
    label_smoothing = 0.1
    PATH_TRAIN = "train_data/"
    PATH_CSV = "Files/"
    OUTPUT_DIR_LOG = "logs/"
    OUTPUT_DIR_MODEL = "models/"

    experiment_name =f"{model_name}_{color_mode}_lr_{lr}_focal{focal_type}_g{focal_gamma}_resize{size}"

if CFG.debug:
    CFG.epochs = 5
    train_kaggle_public = train_kaggle_public[:1500]
    CFG.print_freq = 20