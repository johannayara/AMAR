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
        ## Pad-masked CSI preprocessing (src/data/csi_preprocess.py). All off by default so the raw
        ## pipeline is unchanged. The AMAR_CSI_PREPROCESS env var (e.g. "static,temporal,zscore")
        ## overrides this block for shell-driven ablations.
        "preprocess": {
            "static_removal": False,   # subtract each sample's temporal mean per feature (static multipath)
            "temporal_zscore": False,  # static_removal + divide by the per-feature temporal std
            "per_room_zscore": False,  # standardize each room with its own pad-masked feature mean/std
        },
    },
    #
    ## hyperparameters of models
    "nn": {
        "lr": 0.001162659394854292,                                     # learning rate
        "epoch": 100,                                 # number of epochs 
        "batch_size":32,                              # batch size
        "threshold": 0.5,                               # threshold to binarize sigmoid outputs
        "scheduler": {
            "type": "cosine_warmup",  # type of scheduler
            "num_warmup_epochs": 1,  # number of warmup epochs
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
        "weight_decay": 2e-4,
        ## Channel normalization inside the THAT Encoder CNN branches: "batch" (default, unchanged),
        ## "group" (GroupNorm; BatchNorm running stats are estimated on the training rooms and shift
        ## under domain shift, so GroupNorm is the room-agnostic choice), or "layer" (GroupNorm with a
        ## single group). AMAR_ENCODER_NORM overrides this from the environment.
        "encoder_norm": "batch",
        ## Feature extractor for the CSI models.
        ##   "default": DensityMapNet, SetLocalizer (density head="coords") and the room/H-WILD
        ##     models use THAT (src/models/bce_that.py); AMAR_WO_RVQ uses its CNN Backbone
        ##     (src/models/modules/molecules.py). Unchanged behaviour.
        ##   "thatv": DensityMapNet, SetLocalizer and AMAR_WO_RVQ use THATV
        ##     (src/models/that_variance.py) — a THAT-derived encoder that keeps per-time-step tokens,
        ##     injects per-feature temporal statistics (mean/std/first-difference = CSI variance over
        ##     time) and pools by attention instead of a global sum. Aimed at localization; see the
        ##     module docstring.
        ##   "thatv-nostats" / "thatv-notokens" / "thatv-nopool": THATV with the named feature
        ##     disabled, for ablating which addition helps or hurts. Comma lists work, e.g.
        ##     "thatv-nostats,nopool". "thatv-notokens" restores THAT's valid-conv geometry and is
        ##     single-vector-only (forward_sequence raises), so it cannot back AMAR_WO_RVQ; the
        ##     density and coords heads call forward(), so they accept it.
        ##   "cnn": the AMAR CNN Backbone (src/models/modules/molecules.py) — currently DensityMapNet
        ##     only. It emits a token sequence (B, N, D) instead of one vector, so DensityMapNet sums
        ##     it over time to (B, D) to match THAT's aggregation. ~5.7x fewer parameters than THAT
        ##     (0.19M vs 1.08M). Other models ignore it and fall back to THAT.
        "backbone": "default",
        "num_obj_queries": 10, #TODO: change this if more people
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
        "sigma": 0.08,
        "count_loss_weight": 1.0,
        "peak_threshold": 0.25,  # fraction of the map max used to extract predicted locations
        ## How the reported count is decided.
        ##   "layout_free" (default): count = number of predicted map peaks above a validation-
        ##     calibrated absolute threshold; the map-mass count and the position error are reported
        ##     alongside. Needs no knowledge of the room's known locations at inference time.
        ##   "template": the original rule, count = number of the room's known locations whose
        ##     per-location occupancy readout clears the calibrated threshold. Kept for reproducing
        ##     the earlier protocol.
        "count_rule": "layout_free",
        ## Which head the model uses.
        ##   "density" (default): the continuous H x W density map; the count is read off the map
        ##     (layout_free peaks, or the per-location template readout).
        ##   "coords": a single head that directly regresses the user locations (x, y) and counts
        ##     them -- count = number of predicted locations, reported in meters. Uses SetLocalizer
        ##     (src/models/set_localization.py) with the Hungarian set loss.
        "head": "coords",
        ## Absolute floor for peak extraction (e.g. the pcap-inference path, which has no validation
        ## split to calibrate an empty gate on). A map whose maximum is below it yields no peaks.
        "peak_abs_floor": 0.0,
        ## Class balance: WiMANS has only 5.3% empty-room (count-0) frames per room (99 of 1881),
        ## and they are identical across rooms, so a plain shuffle lets the model ignore the empty
        ## case. Draw the training batches with class-balanced weights (every count class equally
        ## likely) so the empty-room frames are seen as often as the crowded ones. No data is
        ## discarded, unlike undersampling the majority classes.
        "balance_empty_class": True,
        ## Density head size. A large decoder can satisfy the source-room loss by emitting that
        ## room's marginal map and ignoring the input, which does not transfer to another room, so the
        ## decoder is kept small and regularised.
        "decoder_hidden": 64,
        "decoder_dropout": 0.3,
        ## Occupancy decision threshold (stage 2 of the count decision). None => calibrate it on the
        ## validation split with the balanced objective (best for a final run, but optimistic when the
        ## same split is then scored). Set a fixed value (e.g. 0.5) for hyperparameter search, so a
        ## config cannot win by fitting the threshold sweep rather than the model.
        "eval_threshold": None,
        ## Two-stage count decision. Stage 1 is an empty gate: a sample whose density-map maximum is
        ## below "empty_threshold" is predicted empty regardless of its per-location readouts. This
        ## decouples "is the room empty" from "how many locations are occupied" and is what stops a
        ## raised map floor under domain shift from turning every empty frame into a count >= 1.
        ## "empty_threshold" None => calibrate the gate on the validation split; set False to disable
        ## the gate entirely (single-threshold behaviour, calibrated on every frame).
        "staged_count": True,
        "empty_threshold": None,
    },

    ## H-WILD human-held-device localization (src/data/hwild.py). "path" is the dataset root holding
    ## the Conference/, Laboratory/, Office/ and Lounge/ folders. Packets are sliced into fixed-length
    ## windows; "window" must be a multiple of 20 and at least 320, which is what the THAT backbone's
    ## pooling/conv geometry requires (see src/models/bce_that.py). The regression head is a small MLP
    ## on the backbone embedding; the output is the normalized (x, y) position.
    "hwild": {
        "path": "dataset/hwild",
        "window": 400,
        "stride": 200,
        "head_hidden": 128,
        "head_dropout": 0.2,
        "epoch": 60,
        "patience": 20,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 2e-4,
    },

    ## Grid used by the LOR per-location visualizations (src/utils.visualize_room_grid_performance):
    ## a room's [0,1]^2 frame is split into viz.grid x viz.grid cells, each cell acting as a
    ## "location" for the standard count/confusion/error figures.
    "viz": {
        "grid": 5,
    },
}

preset["nn"]["num_classes"] = 6 if preset["task"] in ("location", "count") else 10
preset["nn"]["num_count_classes"] = 6  # group-count task: 0..5 people
