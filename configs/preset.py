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
        "epoch": 100,                                 # number of epochs 
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

    ## Room-agnostic normalized coordinates of the 5 WiMANS locations, in a frame shared by every
    ## room: origin at the transmitter (bottom-left corner of the room), x increasing to the right,
    ## y increasing away from the transmitter (up), distances in cm divided by 1000.
    ##
    ## Derived from the dimensions printed on the WiMANS Fig. 2 layouts (all rooms are 510 x 1030 cm,
    ## with the TX-side wall 890 cm tall). Values are the ideal positions, i.e. the red markers in
    ## the figures minus their systematic ~4 cm downward offset. Classroom columns are at 410 / 255 /
    ## 100 cm from the left wall, at 600 (A/D), 455 (C) and 310 (B/E) cm from the TX wall.
    "layouts": {
        "classroom": {
            "a": (0.410, 0.600),
            "b": (0.410, 0.310),
            "c": (0.255, 0.455),
            "d": (0.100, 0.600),
            "e": (0.100, 0.310),
        },
        "meeting_room": {
            "a": (0.480, 0.545),
            "b": (0.480, 0.285),
            "c": (0.255, 0.415),
            "d": (0.030, 0.545),
            "e": (0.030, 0.285),
        },
        "empty_room": {
            "a": (0.370, 0.625),
            "b": (0.370, 0.265),
            "c": (0.255, 0.445),
            "d": (0.140, 0.625),
            "e": (0.140, 0.265),
        },
    },

    ## Density-map group counting: spatial resolution of the map and the width of the Gaussian
    ## placed on each occupied location when building the target.
    "density": {
        "grid_size": 32,
        "sigma": 0.06,
        "count_loss_weight": 1.0,
        "peak_threshold": 0.25,  # fraction of the map max used to extract predicted locations
        ## Class balance: WiMANS has only 5.3% empty-room (count-0) frames per room (99 of 1881),
        ## and they are identical across rooms, so a plain shuffle lets the model ignore the empty
        ## case. Draw the training batches with class-balanced weights (every count class equally
        ## likely) so the empty-room frames are seen as often as the crowded ones. No data is
        ## discarded, unlike undersampling the majority classes.
        "balance_empty_class": True,
    },
}

preset["nn"]["num_classes"] = 6 if preset["task"] in ("location", "count") else 10
preset["nn"]["num_count_classes"] = 6  # group-count task: 0..5 people
