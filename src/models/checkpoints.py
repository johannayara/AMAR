"""
[file]          checkpoints.py
[description]   Uniform checkpoint save/load for every model the WiMANS cross-domain harness trains
                (scripts/run_cross_domain.py), so a saved run can be rebuilt by any consumer -- the
                input-dependence diagnostic, reeval, or live inference -- without the consumer knowing
                which architecture produced it.

                The density head already had save_density_checkpoint / load_density_checkpoint in
                src/models/density_map.py; this module adds the same capability for the coordinate head
                (SetLocalizer), AMAR_WO_RVQ and the dummy prior baseline, and one dispatcher
                (load_cross_domain_model) that reads the checkpoint's own "model_name" and rebuilds the
                right module. The dispatcher imports the model modules lazily, so importing this module
                from a model module (to save) does not create an import cycle.

                Every checkpoint written here carries:
                    format          fixed tag, so a foreign .pth fails with a clear message
                    model_name      dispatcher key ("density_map" | "density_map_dem" |
                                    "density_map_coords" | "AMAR_WO_RVQ" | "dummy_model")
                    model_state_dict
                    x_shape         the CSI window shape the model was built for
                    train_envs      rooms the run trained on (a room-agnostic model's domain)
                    backbone, encoder_norm   the encoder config (not recoverable from weights alone)
                    ctor            dict of the remaining constructor kwargs, enough to rebuild the
                                    module exactly
"""
#
##

import os

import torch

from configs.preset import preset
from src.models.bce_that import resolve_encoder_norm

CHECKPOINT_FORMAT = "amar.cross_domain.checkpoint"


#
## ---------------------------------------------------------------------------------------------- ##
## --------------------------------------- saving ----------------------------------------------- ##
#
def _base(var_model_name, var_model, var_x_shape, var_train_envs, var_ctor):
    """
    [description]
    : the metadata every cross-domain checkpoint carries; the model-specific savers add their ctor
      kwargs through var_ctor.
    """
    return {
        "format": CHECKPOINT_FORMAT,
        "model_name": var_model_name,
        "model_state_dict": var_model.state_dict(),
        "x_shape": tuple(var_x_shape),
        "train_envs": list(var_train_envs),
        "backbone": preset["nn"].get("backbone", "default"),
        "encoder_norm": resolve_encoder_norm(),
        "ctor": dict(var_ctor),
    }


def _write(var_checkpoint, var_save_path, var_tag="model"):
    os.makedirs(str(var_save_path), exist_ok=True)
    var_out_path = os.path.join(str(var_save_path), f"{var_tag}.pth")
    torch.save(var_checkpoint, var_out_path)
    print(f"  Checkpoint saved to {var_out_path} (model_name={var_checkpoint['model_name']})")
    return var_out_path


def save_coord_checkpoint(var_model, var_x_shape, var_train_envs, var_save_path, var_tag="model"):
    """
    [description]
    : save the coordinate head (SetLocalizer). The constructor kwargs are introspected from the live
      module rather than read from the preset, so a model built with non-preset sizes (the runner
      hardcodes embedding_dim=100) still reloads exactly.
    """
    var_layer = var_model.decoder.layers[0]
    var_ctor = {
        "embedding_dim": var_model.memory_proj.in_features,
        "d_model": var_model.memory_proj.out_features,
        "num_queries": var_model.num_queries,
        "num_decoder_layers": len(var_model.decoder.layers),
        "nhead": var_layer.self_attn.num_heads,
        "dim_feedforward": var_layer.linear1.out_features,
        "dropout": var_layer.dropout.p,
    }
    return _write(_base("density_map_coords", var_model, var_x_shape, var_train_envs, var_ctor),
                  var_save_path, var_tag)


def save_amar_checkpoint(var_model, var_x_shape, var_train_envs, var_save_path, var_tag="model"):
    """
    [description]
    : save AMAR_WO_RVQ. The decoder's size (d_model, depth, queries, ffn, heads, classes) is
      introspected from the live module so the checkpoint reloads exactly even if the preset changed
      after the model was built; temp_cross / embedding_time_dim are not stored on the module (the
      decoder layer never reads temp_cross_attention), so they come from the preset.
    """
    var_decoder = var_model.decoder
    var_ctor = {
        "features_dim": var_decoder.d_model,
        "embedding_time_dim": preset["nn"]["token_length"],
        "num_decoder_layers": len(var_decoder.decoder_layers),
        "temp_cross": preset["nn"]["cross_attention_temp"],
        "n_attention_heads": var_decoder.nhead,
        "num_queries": var_decoder.query_embed.shape[0],
        "dim_feedforward": var_decoder.decoder_layers[0].ffn[0].out_features,
        "query_dropout_rate": var_decoder.query_dropout_rate,
        "num_classes": var_decoder.class_embed.out_features,
    }
    return _write(_base("AMAR_WO_RVQ", var_model, var_x_shape, var_train_envs, var_ctor),
                  var_save_path, var_tag)


def save_dummy_checkpoint(var_model, var_x_shape, var_y_shape, var_train_envs, var_save_path,
                          var_tag="model"):
    """
    [description]
    : save the dummy prior baseline (a single linear map).
    """
    return _write(_base("dummy_model", var_model, var_x_shape, var_train_envs,
                        {"y_shape": [int(var_y_shape)]}),
                  var_save_path, var_tag)


#
## ---------------------------------------------------------------------------------------------- ##
## --------------------------------------- loading ---------------------------------------------- ##
#
def _load_coord(var_checkpoint, var_device):
    from src.models.set_localization import SetLocalizer
    var_model = SetLocalizer(tuple(var_checkpoint["x_shape"]), **var_checkpoint["ctor"]).to(var_device)
    var_model.load_state_dict(var_checkpoint["model_state_dict"])
    var_model.eval()
    return var_model


def _load_amar(var_checkpoint, var_device):
    from src.models.AMAR_WO_RVQ import AMAR_WO_RVQ
    var_model = AMAR_WO_RVQ(tuple(var_checkpoint["x_shape"]), **var_checkpoint["ctor"]).to(var_device)
    var_model.load_state_dict(var_checkpoint["model_state_dict"])
    var_model.eval()
    return var_model


def _load_dummy(var_checkpoint, var_device):
    from src.models.dummy_model import DUMMY
    var_model = DUMMY(tuple(var_checkpoint["x_shape"]), var_checkpoint["ctor"]["y_shape"]).to(var_device)
    var_model.load_state_dict(var_checkpoint["model_state_dict"])
    var_model.eval()
    return var_model


def checkpoint_model_name(var_checkpoint):
    """
    [description]
    : the dispatcher key of a checkpoint. Written checkpoints carry "model_name"; a density checkpoint
      from before this module existed is recognised by its grid_size key.
    """
    var_name = var_checkpoint.get("model_name")
    if var_name:
        return var_name
    if "grid_size" in var_checkpoint:
        return "density_map_dem" if var_checkpoint.get("output_mode") == "raw" else "density_map"
    raise ValueError("checkpoint has no model_name and is not a density checkpoint; "
                     "cannot tell which architecture it describes")


def load_cross_domain_model(var_path, var_device=None):
    """
    [description]
    : rebuild the model a cross-domain checkpoint describes and load its weights, dispatching on the
      checkpoint's own model_name.
    : return: (model in eval mode, checkpoint dict)
    """
    if var_device is None:
        from src.utils import select_device
        var_device = select_device()
    var_checkpoint = torch.load(var_path, map_location=var_device)
    var_name = checkpoint_model_name(var_checkpoint)
    if var_name in ("density_map", "density_map_dem"):
        from src.models.density_map import load_density_checkpoint
        return load_density_checkpoint(var_path, var_device)
    if var_name == "density_map_coords":
        return _load_coord(var_checkpoint, var_device), var_checkpoint
    if var_name == "AMAR_WO_RVQ":
        return _load_amar(var_checkpoint, var_device), var_checkpoint
    if var_name == "dummy_model":
        return _load_dummy(var_checkpoint, var_device), var_checkpoint
    raise ValueError(f"unknown checkpoint model_name {var_name!r}")
