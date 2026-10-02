import os
"""
[file]          preset.py
[description]   default settings of WiFi-based models
"""
#
##
preset = {
    #
    ## define model
    "model": "AMAR_WO_RVQ",                                    #  BCE_ABLSTM, BCE_THAT, DEM_ABLSTM
                                                              #  DEM_THAT",  AMAR, AMAR_WO_RVQ
                                                            
    # "model": "MLP",
    ## define task
    "task": "location",                                 #  "activity"
    #
    ## number of repeated experiments
    "repeat": 8,
    ## path of data
    "path": {
        "data_x": "dataset/wifi_csi/amp",  # directory of CSI amplitude files
        "data_y": "dataset/annotation.csv",  # path of annotation file
        "save": "results/result.json"                           # path to save results
    },
    #
    ## data selection for experiments
    "data": {
        "num_users": ["0","1", "2", "3", "4", "5"] ,   # TODO: fix this number for my app. select number(s) of users, (e.g., ["0", "1"], ["2", "3", "4", "5"])
        "wifi_band": ["5"],                           # select WiFi band(s) (e.g., ["2.4"], ["5"], ["2.4", "5"])
        "environment": ["empty_room", "meeting_room", "classroom"],               # select environment(s) (e.g., ["classroom"], ["meeting_room"], ["empty_room"])
        "length": 3000,                                 # default length of CSI
    },
    #
    ## hyperparameters of models
    "nn": {
        "lr": 5e-4,                                     # learning rate
        "epoch": 20,                                 # number of epochs 
        "batch_size":16,                              # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
        "scheduler": {
            "type": "cosine_warmup",  # type of scheduler
            "num_warmup_epochs": 3,  # number of warmup epochs
            "min_lr_ratio": 0.1  # minimum learning rate ratio
        },
        # Loss function parameters
        "loss": {
            "cost_class_weight": 1.0,  # weight for classification cost
            "aux_loss_weight": 0.25,  # weight for auxiliary losses
            "label_smoothing": 0.2,  # label smoothing factor
            "class_imbalance_weight": 0.25
        },
        "cross_attention_temp": 1,
        "weight_decay": 1e-4,
        "num_obj_queries": 6, #TODO: change this if more people
        "num_decoder_layers":6,
        "dim_FFN": 512,
        "token_length": 188, 
        "d_embedding": 64,
        "n_layers_encoder":4,
        "n_attention_heads": 4,
        "query_dropout_rate": 0,
        "commitment_cost": 0.5,
        "num_codes": 16,
        "num_rvq_layers":4,  # Number of RVQ layers 
        "quantization_dropout": 0.3,  # Dropout rate for quantization layers

},
    

    ## encoding of activities and locations
    "encoding": {
        "activity": {                                   # encoding of different activities
            "nan":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nothing":  [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "walk":     [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "rotation": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "jump":     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "wave":     [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "lie_down": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "pick_up":  [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        },
        "location": {                                   # encoding of different locations
            "nan":  [0, 0, 0, 0, 0],
            "a":    [1, 0, 0, 0, 0],
            "b":    [0, 1, 0, 0, 0],
            "c":    [0, 0, 1, 0, 0],
            "d":    [0, 0, 0, 1, 0],
            "e":    [0, 0, 0, 0, 1],
        },
    },
    "pretrained_path": None,
    # "pretrained_path": "/saved_models/jepa_ssl_empty_room+classroom+meeting_room_20250721_124829/jepa_ssl_final_empty_room+classroom+meeting_room_20250721_124829.pth",
    "transfer_scenario": "freeze_encoder",  # One of ["full", "feature_extractor", "feature_encoder"]
    "save_model": False,  # Whether to save model components
    "saving_path": "./multi_modal_CSI/results/checkpoints/",

    ## Room-agnostic normalized coordinates (x right, y up, origin tx) of the 5 WiMANS
    ## locations, taken from the environment layouts in WiMANS Fig. 2 (all rooms are 510x1030 cm).
    ## This is the shared frame the density map is predicted in, so a location means the same
    ## physical spot in every room.
    "layouts": {
        "classroom": {
            "a": (0.410, 0.600),
            "b": (0.410, 0.165),
            "c": (0.255, 0.455),
            "d": (0.100, 0.600),
            "e": (0.100, 0.165),
        },
        "meeting_room": {
            "a": (0.9401, 0.3376),
            "b": (0.9389, 0.5855),
            "c": (0.4991, 0.4617),
            "d": (0.0571, 0.3379),
            "e": (0.0577, 0.5855),
        },
        "empty_room": {
            "a": (0.7244, 0.2614),
            "b": (0.7232, 0.6045),
            "c": (0.4991, 0.4331),
            "d": (0.2728, 0.2617),
            "e": (0.2734, 0.6045),
        },
    },

    ## Density-map group counting: spatial resolution of the map and the width of the Gaussian
    ## placed on each occupied location when building the target.
    "density": {
        "grid_size": 32,
        "sigma": 0.06,
        "count_loss_weight": 1.0,
        "peak_threshold": 0.25,  # fraction of the map max used to extract predicted locations
    },
}

preset["nn"]["num_classes"] = 6 if preset["task"] in ("location", "count") else 10
