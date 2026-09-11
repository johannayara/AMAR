#
##
import os
import math
import time
import torch
import gc
import numpy as np
from sklearn.model_selection import train_test_split
#
import torch.nn as nn
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from src.models.modules.molecules import Backbone, Transformer_Encoder, TransformerDecoder
from src.models.losses.supervised_loss import HungarianMatchingLoss
from src.train import train
from configs.preset import preset
from src.utils import *
import wandb




class AMAR_WO_RVQ(nn.Module):
    def __init__(self, var_x_shape, features_dim = 20, embedding_time_dim=100, num_decoder_layers=12,
                 temp_cross=1, n_attention_heads=2, num_queries=5, dim_feedforward=1024, query_dropout_rate=0.0, num_classes=10):
        super().__init__()
        # self.feature_extractor = CNNFeatureExtractor(input_channels=var_x_shape[-1], output_channels=features_dim,embedding_time_dim=embedding_time_dim)
        self.feature_extractor = Backbone(input_channels=270, output_channels=preset["nn"]["d_embedding"])
                                                     # embedding_time_dim=preset["cnn_embedding_time_dim"])

        # self.encoder = Transformer_Encoder(var_embedding_shape, num_attention_heads=n_attention_heads,
        #                                    num_transformer_encoder_layers=8)
        self.encoder = Transformer_Encoder( d_model=preset["nn"]["d_embedding"], nhead=n_attention_heads, num_layers=preset["nn"]["n_layers_encoder"],
                 max_total_tokens=preset["nn"]["token_length"])
        self.decoder = TransformerDecoder(
            d_model=features_dim,
            nhead=n_attention_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            num_queries=num_queries,
            temp_cross_attention=temp_cross, 
            query_dropout_rate=query_dropout_rate,
            num_classes= num_classes
        )
        self.decoder.memory_pos_encoding = self.encoder.pos_encoder
    def forward(self, x):

        extracted_features = self.feature_extractor(x)

        memory = self.encoder(extracted_features)

        outputs_class = self.decoder(memory)

        return outputs_class

def run_AMAR_WO_RVQ(data_train_x,
                     data_train_y,
                     data_test_x,
                     data_test_y,
                     var_repeat=10, var_task = "activity", var_env = "empty_room", save_path = "./visualizations/temp"):
    """
    [description]
    : run WiFi-based model Transformer_Encoder_DECODER
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    # Update device selection to check for CUDA first, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    data_valid_x, data_test_x, data_valid_y, data_test_y = train_test_split(data_test_x, data_test_y,
                                                                            test_size=0.5,
                                                                            shuffle=True,
                                                                            random_state=39)
    data_valid_x = data_valid_x.reshape(data_valid_x.shape[0], data_valid_x.shape[1], -1)
    data_train_x = data_train_x.reshape(data_train_x.shape[0], data_train_x.shape[1], -1)
    data_test_x = data_test_x.reshape(data_test_x.shape[0], data_test_x.shape[1], -1)
    #
    data_x_mean = np.mean(data_train_x, axis=1)
    ## shape for model
    var_x_shape = data_train_x[0].shape
    #
    data_train_set = TensorDataset(torch.from_numpy(data_train_x), torch.from_numpy(data_train_y))
    data_valid_set = TensorDataset(torch.from_numpy(data_valid_x), torch.from_numpy(data_valid_y))

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    ## per env results 
    env_result_accuracy = {}
    env_result_total_error = {}
    env_result_ppp = {}
    env_result_precision = {}
    env_result_recall = {}
    env_result_f1_score = {}

    #
    var_macs, var_params = get_model_complexity_info(AMAR_WO_RVQ(var_x_shape,
                                n_attention_heads=preset["nn"]["n_attention_heads"],
                                features_dim=preset["nn"]["d_embedding"],
                                embedding_time_dim=preset["nn"]["token_length"],
                                num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                temp_cross=preset["nn"]["cross_attention_temp"],
                                num_queries=preset["nn"]["num_obj_queries"],
                                dim_feedforward=preset["nn"]["dim_FFN"],
                                query_dropout_rate=preset["nn"]["query_dropout_rate"],
                                num_classes=preset["nn"]["num_classes"]),var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #

    for var_r in range(var_repeat):
        #
        ##
        var_mode = "multi_head"
        name_run = "Empty"
        if preset["pretrained_path"]:
            name_run = f"AMAR_{var_r}_" + "_".join(var_env) + "_" + preset["transfer_scenario"]
        else:
            pretrained_state = "NPT"
            name_run = f"AMAR_{var_r}_" + "_".join(var_env) + "_" + pretrained_state 
        print("Repeat", var_r)
        run = wandb.init(
            project="FINAL_AMAR",
            name= name_run, 
            config=preset,
            reinit=True  
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_AMAR_WO_RVQ = AMAR_WO_RVQ(var_x_shape,
                            n_attention_heads=preset["nn"]["n_attention_heads"],
                            features_dim=preset["nn"]["d_embedding"],
                            embedding_time_dim=preset["nn"]["token_length"],
                            num_decoder_layers=preset["nn"]["num_decoder_layers"],
                            temp_cross=preset["nn"]["cross_attention_temp"],
                            num_queries=preset["nn"]["num_obj_queries"],
                            dim_feedforward=preset["nn"]["dim_FFN"],
                            # pca_embeddings=pca_components
                            query_dropout_rate=preset["nn"]["query_dropout_rate"],
                            num_classes=preset["nn"]["num_classes"]
                            ).to(device)
        

        model_AMAR_WO_RVQ.feature_extractor = torch.compile(model_AMAR_WO_RVQ.feature_extractor)


        optimizer = torch.optim.Adam(model_AMAR_WO_RVQ.parameters(),
                                        lr=preset["nn"]["lr"],
                                        weight_decay=preset["nn"]["weight_decay"])

        loss = HungarianMatchingLoss(
            cost_class_weight=preset["nn"]["loss"]["cost_class_weight"],
            aux_loss_weight=preset["nn"]["loss"]["aux_loss_weight"],
            label_smoothing=preset["nn"]["loss"]["label_smoothing"],
            class_imbalance_weight=preset["nn"]["loss"]["class_imbalance_weight"],
            num_classes=preset["nn"]["num_classes"]
        )
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model=model_AMAR_WO_RVQ,
                                optimizer=optimizer,
                                loss=loss,
                                data_train_set=data_train_set,
                                data_valid_set=data_valid_set,
                                var_threshold=preset["nn"]["threshold"],
                                var_batch_size=preset["nn"]["batch_size"],
                                var_epochs=preset["nn"]["epoch"],
                                device=device,
                                var_mode=var_mode)
        # Save model components
        if preset.get("save_model"):
            save_model_components(preset, model_AMAR_WO_RVQ)
        #
        var_time_1 = time.time()
        #


        ## ---------------------------------------- Test ------------------------------------------
        #
        model_AMAR_WO_RVQ.load_state_dict(var_best_weight)
        #
        test_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y)),
            batch_size=preset["nn"]["batch_size"], shuffle=False
        )
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                preds.append(model_AMAR_WO_RVQ(xb.to(device)).cpu())
        predict_test_y = torch.cat(preds, dim=1).numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        #
        ##

        layers_idxs = ["layer_"+str(i) for i in range(preset["nn"]["num_decoder_layers"])]
        last_layer_only = True
        # Store results for each layer
        all_layers_results = {}
        dict_layer_acc = performance_metrics(data_test_y, predict_test_y, var_mode=var_mode)

        # Process each layer separately
        for idx, layer_idx in enumerate(layers_idxs):
            layer_metrics = dict_layer_acc[layer_idx]
            if var_r == 0:  # Initialize lists on first repeat
                result_ppp.append([])
                result_time_train.append([])
                result_time_test.append([])
                result_total_error.append([])
                result_precision.append([])
                result_recall.append([])
                result_f1_score.append([])
                result_avg_count_error.append([])
                result_accuracy.append([])
            result_accuracy[idx].append(layer_metrics['accuracy'])
            result_ppp[idx].append(layer_metrics['perfect_prediction_percentage'])
            result_time_train[idx].append(var_time_1 - var_time_0)
            result_time_test[idx].append(var_time_2 - var_time_1)
            result_total_error[idx].append(layer_metrics['total_error'])
            result_precision[idx].append(layer_metrics['precision'])
            result_recall[idx].append(layer_metrics['recall'])
            result_f1_score[idx].append(layer_metrics['f1_score'])
            result_avg_count_error[idx].append(layer_metrics['mean_count_error'])

        if last_layer_only:
            layer_metrics = dict_layer_acc["layer_" +str(preset["nn"]["num_decoder_layers"] - 1)]
            wandb.log({
                f"test_results/repeat": var_r,
                f"test_results/train_time": var_time_1 - var_time_0,
                f"test_results/test_time": var_time_2 - var_time_1,
                f"test_results/TOTAL_TESTSET_ERROR": layer_metrics['total_error'],
                f"test_results/TOTAL_TESTSET_perfect_prediction_percentage": layer_metrics[
                    'perfect_prediction_percentage'],
                f"test_results/TOTAL_ACCURACY": layer_metrics['accuracy'],
                f"test_results/mean_count_error": layer_metrics['mean_count_error'],
                f"test_results/error_per_person_1": layer_metrics['error_per_person'][0],
                f"test_results/error_per_person_2": layer_metrics['error_per_person'][1],
                f"test_results/error_per_person_3": layer_metrics['error_per_person'][2],
                f"test_results/error_per_person_4": layer_metrics['error_per_person'][3],
                f"test_results/error_per_person_5": layer_metrics['error_per_person'][4],
                f"test_results/precision": layer_metrics['precision'],
                f"test_results/recall": layer_metrics['recall'],
                f"test_results/f1_score": layer_metrics['f1_score']
            }, step=var_r + 100000)

            print(
                  "- Total Error %.6f" % layer_metrics['total_error'],
                  "- Perfect Prediction Percentage %.6f" % layer_metrics['perfect_prediction_percentage'])
        
        del optimizer, loss
        torch.cuda.empty_cache()
        gc.collect()
        if var_r == var_repeat - 1:
            last_model = model_AMAR_WO_RVQ   
        else:
            del model_AMAR_WO_RVQ
    # Calculate averages and standard errors for each layer
    for layer_idx_num, layer_idx in enumerate(layers_idxs):
        # Calculate metrics with standard errors
        ppp_array = np.array(result_ppp[layer_idx_num])
        precision_array = np.array(result_precision[layer_idx_num])
        recall_array = np.array(result_recall[layer_idx_num])
        f1_array = np.array(result_f1_score[layer_idx_num])
        accuracy_array = np.array(result_accuracy[layer_idx_num])
        total_error_array = np.array(result_total_error[layer_idx_num])
        
        # Store results for this layer
        all_layers_results[layer_idx] = {
            'avg_PPP': float(np.mean(ppp_array)),
            'avg_precision': float(np.mean(precision_array)),
            'avg_recall': float(np.mean(recall_array)),
            'avg_f1_score': float(np.mean(f1_array)),
            'avg_accuracy': float(np.mean(accuracy_array)),
            'avg_total_error': float(np.mean(total_error_array)),
            'std_PPP': float(np.std(ppp_array, ddof=1)) if len(ppp_array) > 1 else 0.0,
            'std_precision': float(np.std(precision_array, ddof=1)) if len(precision_array) > 1 else 0.0,
            'std_recall': float(np.std(recall_array, ddof=1)) if len(recall_array) > 1 else 0.0,
            'std_f1_score': float(np.std(f1_array, ddof=1)) if len(f1_array) > 1 else 0.0,
            'std_accuracy': float(np.std(accuracy_array, ddof=1)) if len(accuracy_array) > 1 else 0.0,
            'std_total_error': float(np.std(total_error_array, ddof=1)) if len(total_error_array) > 1 else 0.0,
            'se_PPP': float(np.std(ppp_array, ddof=1) / np.sqrt(len(ppp_array))) if len(ppp_array) > 1 else 0.0,
            'se_precision': float(np.std(precision_array, ddof=1) / np.sqrt(len(precision_array))) if len(precision_array) > 1 else 0.0,
            'se_recall': float(np.std(recall_array, ddof=1) / np.sqrt(len(recall_array))) if len(recall_array) > 1 else 0.0,
            'se_f1_score': float(np.std(f1_array, ddof=1) / np.sqrt(len(f1_array))) if len(f1_array) > 1 else 0.0,
            'se_accuracy': float(np.std(accuracy_array, ddof=1) / np.sqrt(len(accuracy_array))) if len(accuracy_array) > 1 else 0.0,
            'se_total_error': float(np.std(total_error_array, ddof=1) / np.sqrt(len(total_error_array))) if len(total_error_array) > 1 else 0.0
        }
        
        wandb.log({
            f"test_results/{layer_idx}/avg_PPP": all_layers_results[layer_idx]['avg_PPP'],
            f"test_results/{layer_idx}/avg_train_time": sum(result_time_train[layer_idx_num]) / len(result_time_train[layer_idx_num]),
            f"test_results/{layer_idx}/avg_test_time": sum(result_time_test[layer_idx_num]) / len(result_time_test[layer_idx_num]),
            f"test_results/{layer_idx}/avg_total_error": all_layers_results[layer_idx]['avg_total_error'],
            f"test_results/{layer_idx}/avg_precision": all_layers_results[layer_idx]['avg_precision'],
            f"test_results/{layer_idx}/avg_recall": all_layers_results[layer_idx]['avg_recall'],
            f"test_results/{layer_idx}/avg_f1_score": all_layers_results[layer_idx]['avg_f1_score'],
            f"test_results/{layer_idx}/avg_count_error": sum(result_avg_count_error[layer_idx_num]) / len(result_avg_count_error[layer_idx_num]),
            f"test_results/{layer_idx}/avg_accuracy": all_layers_results[layer_idx]['avg_accuracy']
        })  # Use an even larger offset for averages
    
    # Use the last layer for visualization and final results
    last_layer = layers_idxs[-1]
    last_layer_predictions = predict_test_y[last_layer] if isinstance(predict_test_y, dict) else predict_test_y
    # dict_true_acc = all_layers_results[last_layer]

    # Run visualization with the last layer's predictions
    
    log_random_attention_weights_final(last_model, np.argmax(predict_test_y[-1], axis=-1), np.argmax(data_test_y, axis=-1), 1000000000, 50, var_task)
    
    viz_stats = visualize_model_performance(
        y_pred=last_layer_predictions,
        y_true=data_test_y,
        var_mode=var_mode,
        save_dir=save_path
    )
    print("\nDetailed Performance Analysis:")
    print(f"Mean Error: {viz_stats['mean_error']:.4f} ± {viz_stats['error_std']:.4f}")
    print("\nClass-wise Mean Absolute Error:")
    for i, error in enumerate(viz_stats['class_wise_mae']):
        print(f"Class {i}: {error:.4f}")
    print(f"\nPerfect Predictions: {viz_stats['perfect_predictions'] * 100:.2f}%")
    wandb.finish()
    return all_layers_results
## ====================================================================================================================
# CROSS DOMAIN 

def run_cross_domain(data_x_train, data_y_train, test_sets_by_env, var_repeat=10, var_task = "activity", var_env = "empty_room", save_path = "./visualizations/temp"):                    
    """
    [description]
    : run WiFi-based model Transformer_Encoder_DECODER
    [parameter]
    : data_train_x: numpy array, CSI amplitude to train model
    : data_train_y: numpy array, labels to train model
    : data_test_x: numpy array, CSI amplitude to test model
    : data_test_y: numpy array, labels to test model
    : var_repeat: int, number of repeated experiments
    [return]
    : result: dict, results of experiments
    """
    #
    ##
    ## ============================================ Preprocess ============================================
    #
    # Update device selection to check for CUDA first, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    data_x_train, data_x_valid, data_y_train, data_y_valid = train_test_split(
        data_x_train, data_y_train,
        test_size=0.1, 
        shuffle=True,
        random_state=39
    )
    data_x_valid = data_x_valid.reshape(data_x_valid.shape[0], data_x_valid.shape[1], -1)
    data_x_train = data_x_train.reshape(data_x_train.shape[0], data_x_train.shape[1], -1)
    var_x_shape = data_x_train[0].shape

    data_train_set = TensorDataset(torch.from_numpy(data_x_train), torch.from_numpy(data_y_train))
    data_valid_set = TensorDataset(torch.from_numpy(data_x_valid), torch.from_numpy(data_y_valid))

    #
    ##
    ## ========================================= Train & Evaluate =========================================
    ## per env results 
    env_result_accuracy = {}
    env_result_total_error = {}
    env_result_ppp = {}
    env_result_precision = {}
    env_result_recall = {}
    env_result_f1_score = {}

    #
    var_macs, var_params = get_model_complexity_info(AMAR_WO_RVQ(var_x_shape,
                                n_attention_heads=preset["nn"]["n_attention_heads"],
                                features_dim=preset["nn"]["d_embedding"],
                                embedding_time_dim=preset["nn"]["token_length"],
                                num_decoder_layers=preset["nn"]["num_decoder_layers"],
                                temp_cross=preset["nn"]["cross_attention_temp"],
                                num_queries=preset["nn"]["num_obj_queries"],
                                dim_feedforward=preset["nn"]["dim_FFN"],
                                query_dropout_rate=preset["nn"]["query_dropout_rate"],
                                num_classes=preset["nn"]["num_classes"]),var_x_shape, as_strings=False)
    print("Parameters:", var_params, "- FLOPs:", var_macs * 2)

    #

    for var_r in range(var_repeat):
        #
        ##
        var_mode = "multi_head"
        name_run = "Empty"
        if preset["pretrained_path"]:
            name_run = f"AMAR_{var_r}_" + "_".join(var_env) + "_" + preset["transfer_scenario"]
        else:
            pretrained_state = "NPT"
            name_run = f"AMAR_{var_r}_" + "_".join(var_env) + "_" + pretrained_state 
        print("Repeat", var_r)
        run = wandb.init(
            project="FINAL_AMAR",
            name= name_run, 
            config=preset,
            reinit=True  
        )
        #
        torch.random.manual_seed(var_r + 39)
        #
        model_AMAR_WO_RVQ = AMAR_WO_RVQ(var_x_shape,
                            n_attention_heads=preset["nn"]["n_attention_heads"],
                            features_dim=preset["nn"]["d_embedding"],
                            embedding_time_dim=preset["nn"]["token_length"],
                            num_decoder_layers=preset["nn"]["num_decoder_layers"],
                            temp_cross=preset["nn"]["cross_attention_temp"],
                            num_queries=preset["nn"]["num_obj_queries"],
                            dim_feedforward=preset["nn"]["dim_FFN"],
                            # pca_embeddings=pca_components
                            query_dropout_rate=preset["nn"]["query_dropout_rate"],
                            num_classes=preset["nn"]["num_classes"]
                            ).to(device)
        

        model_AMAR_WO_RVQ.feature_extractor = torch.compile(model_AMAR_WO_RVQ.feature_extractor)


        optimizer = torch.optim.Adam(model_AMAR_WO_RVQ.parameters(),
                                        lr=preset["nn"]["lr"],
                                        weight_decay=preset["nn"]["weight_decay"])

        loss = HungarianMatchingLoss(
            cost_class_weight=preset["nn"]["loss"]["cost_class_weight"],
            aux_loss_weight=preset["nn"]["loss"]["aux_loss_weight"],
            label_smoothing=preset["nn"]["loss"]["label_smoothing"],
            class_imbalance_weight=preset["nn"]["loss"]["class_imbalance_weight"],
            num_classes=preset["nn"]["num_classes"]
        )
        var_time_0 = time.time()
        #
        ## ---------------------------------------- Train -----------------------------------------
        #
        var_best_weight = train(model=model_AMAR_WO_RVQ,
                                optimizer=optimizer,
                                loss=loss,
                                data_train_set=data_train_set,
                                data_valid_set=data_valid_set,
                                var_threshold=preset["nn"]["threshold"],
                                var_batch_size=preset["nn"]["batch_size"],
                                var_epochs=preset["nn"]["epoch"],
                                device=device,
                                var_mode=var_mode)
        # Save model components
        if preset.get("save_model"):
            save_model_components(preset, model_AMAR_WO_RVQ)
        #
        var_time_1 = time.time()
        #


        ## ---------------------------------------- Test ------------------------------------------
        #
        model_AMAR_WO_RVQ.load_state_dict(var_best_weight)
        #
        test_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.from_numpy(data_test_x), torch.from_numpy(data_test_y)),
            batch_size=preset["nn"]["batch_size"], shuffle=False
        )
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                preds.append(model_AMAR_WO_RVQ(xb.to(device)).cpu())
        predict_test_y = torch.cat(preds, dim=1).numpy()
        #
        var_time_2 = time.time()
        #
        ## -------------------------------------- Evaluate ----------------------------------------
        # Per-env evaluation: 
        if test_sets_by_env:
            for env_name, (X_env, y_env) in test_sets_by_env.items():
                X_env_reshaped = X_env.reshape(X_env.shape[0], X_env.shape[1], -1)
                env_loader = torch.utils.data.DataLoader(
                    TensorDataset(torch.from_numpy(X_env_reshaped), torch.from_numpy(y_env)),
                    batch_size=preset["nn"]["batch_size"], shuffle=False
                )
                env_preds = []
                with torch.no_grad():
                    for xb, _ in env_loader:
                        env_preds.append(model_AMAR_WO_RVQ(xb.to(device)).cpu())
                env_predict_y = torch.cat(env_preds, dim=1).numpy()

                env_metrics = performance_metrics(y_env, env_predict_y, var_mode=var_mode)
                env_last_layer = env_metrics["layer_" + str(preset["nn"]["num_decoder_layers"] - 1)]
                # Initialize this env's lists on first sighting
                if env_name not in env_result_accuracy:
                    env_result_accuracy[env_name] = []
                    env_result_total_error[env_name] = []
                    env_result_ppp[env_name] = []
                    env_result_precision[env_name] = []
                    env_result_recall[env_name] = []
                    env_result_f1_score[env_name] = []

                env_result_accuracy[env_name].append(env_last_layer['accuracy'])
                env_result_total_error[env_name].append(env_last_layer['total_error'])
                env_result_ppp[env_name].append(env_last_layer['perfect_prediction_percentage'])
                env_result_precision[env_name].append(env_last_layer['precision'])
                env_result_recall[env_name].append(env_last_layer['recall'])
                env_result_f1_score[env_name].append(env_last_layer['f1_score'])

                wandb.log({
                    f"test_results_per_env/{env_name}/accuracy": env_last_layer['accuracy'],
                    f"test_results_per_env/{env_name}/total_error": env_last_layer['total_error'],
                    f"test_results_per_env/{env_name}/perfect_prediction_percentage": env_last_layer['perfect_prediction_percentage'],
                    f"test_results_per_env/{env_name}/precision": env_last_layer['precision'],
                    f"test_results_per_env/{env_name}/recall": env_last_layer['recall'],
                    f"test_results_per_env/{env_name}/f1_score": env_last_layer['f1_score'],
                }, step=var_r + 100000)

                print(f"  [{env_name}] Total Error: {env_last_layer['total_error']:.6f} "
                        f"| Perfect Prediction %: {env_last_layer['perfect_prediction_percentage']:.6f}")
        del optimizer, loss
        torch.cuda.empty_cache()
        gc.collect()
        if var_r == var_repeat - 1:
            last_model = model_AMAR_WO_RVQ   
        else:
            del model_AMAR_WO_RVQ

    all_envs_results = {}
    for env_name in env_result_accuracy.keys():
        acc_arr = np.array(env_result_accuracy[env_name])
        err_arr = np.array(env_result_total_error[env_name])
        ppp_arr = np.array(env_result_ppp[env_name])
        prec_arr = np.array(env_result_precision[env_name])
        rec_arr = np.array(env_result_recall[env_name])
        f1_arr = np.array(env_result_f1_score[env_name])

        def _mean_std_se(arr):
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            se = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            return mean, std, se

        acc_mean, acc_std, acc_se = _mean_std_se(acc_arr)
        err_mean, err_std, err_se = _mean_std_se(err_arr)
        ppp_mean, ppp_std, ppp_se = _mean_std_se(ppp_arr)
        prec_mean, prec_std, prec_se = _mean_std_se(prec_arr)
        rec_mean, rec_std, rec_se = _mean_std_se(rec_arr)
        f1_mean, f1_std, f1_se = _mean_std_se(f1_arr)

        all_envs_results[env_name] = {
            'avg_accuracy': acc_mean, 'std_accuracy': acc_std, 'se_accuracy': acc_se,
            'avg_total_error': err_mean, 'std_total_error': err_std, 'se_total_error': err_se,
            'avg_PPP': ppp_mean, 'std_PPP': ppp_std, 'se_PPP': ppp_se,
            'avg_precision': prec_mean, 'std_precision': prec_std, 'se_precision': prec_se,
            'avg_recall': rec_mean, 'std_recall': rec_std, 'se_recall': rec_se,
            'avg_f1_score': f1_mean, 'std_f1_score': f1_std, 'se_f1_score': f1_se,
        } 
        print(f"\n[{env_name}] avg over {var_repeat} repeats: "
              f"Accuracy {acc_mean:.4f} ± {acc_se:.4f} | "
              f"Total Error {err_mean:.4f} ± {err_se:.4f} | "
              f"PPP {ppp_mean:.4f} ± {ppp_se:.4f}")
    wandb.finish()
    return all_envs_results
