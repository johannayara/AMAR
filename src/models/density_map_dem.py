"""
[file]          density_map_dem.py
[description]   Group counting and localization from WiFi CSI via a continuous spatial density map,
                trained with the DEM objective (direct error minimization) instead of BCE.

                DEM regresses the map directly: the density head has no sigmoid and the objective is a
                smooth-L1 (Huber) loss against the same Gaussian target the BCE model is trained on.
                This mirrors the bce_that / dem_that split in this repo.

                Everything else -- backbone, decoder, kernel buffers, target rendering, per-location
                readout, metrics, visualization, training loop and the three protocols -- is shared
                with src/models/density_map.py, so the two objectives are directly comparable under an
                identical protocol. This module only selects var_loss_mode="dem".
"""
#
##

from src.models.density_map import (
    DensityMapNet,
    run_density_map,
    run_density_map_cross_domain,
    run_density_map_few_shot,
)

#
##
class DensityMapDEMNet(DensityMapNet):
    """
    [description]
    : density-map model with a raw (linear) density head, i.e. the output the DEM smooth-L1 objective
      regresses directly. Identical to DensityMapNet apart from var_output_mode="raw".
    """
    #
    ##
    def __init__(self, *var_args, **var_kwargs):
        var_kwargs.setdefault("var_output_mode", "raw")
        super().__init__(*var_args, **var_kwargs)


#
##
def run_density_map_dem(data_train_x,
                        data_train_y,
                        data_test_x,
                        data_test_y,
                        var_repeat=10, var_task="count", var_env="empty_room",
                        save_path="./visualizations/temp"):
    """
    [description]
    : single-room DEM run. Same protocol and metrics as run_density_map, with the smooth-L1 objective.
    """
    return run_density_map(data_train_x, data_train_y, data_test_x, data_test_y,
                           var_repeat=var_repeat, var_task=var_task, var_env=var_env,
                           save_path=save_path, var_loss_mode="dem")


#
##
def run_density_map_dem_cross_domain(train_sets_by_env,
                                     test_sets_by_env,
                                     var_repeat=10, var_task="count",
                                     save_path="./visualizations/temp"):
    """
    [description]
    : cross-domain / leave-one-room-out DEM run. Same protocol and metrics as
      run_density_map_cross_domain, with the smooth-L1 objective.
    """
    return run_density_map_cross_domain(train_sets_by_env, test_sets_by_env,
                                        var_repeat=var_repeat, var_task=var_task,
                                        save_path=save_path, var_loss_mode="dem")


#
##
def run_density_map_dem_few_shot(data_train_x,
                                 data_train_y,
                                 test_sets_by_env,
                                 var_few_shot_ratio=0.05,
                                 var_kd_weight=1.0,
                                 var_kd_temperature=1.0,
                                 var_teacher_epochs=None,
                                 var_student_epochs=None,
                                 var_compile=True,
                                 var_repeat=10, var_task="count", var_env="empty_room",
                                 save_path="./visualizations/temp"):
    """
    [description]
    : few-shot DEM run. The teacher and student both use the raw head; the distillation term is a
      smooth-L1 match of the two density maps (the temperature-scaled Bernoulli distillation of
      OccupancyDistillationLoss does not apply to a raw head).
    """
    return run_density_map_few_shot(data_train_x, data_train_y, test_sets_by_env,
                                    var_few_shot_ratio=var_few_shot_ratio,
                                    var_kd_weight=var_kd_weight,
                                    var_kd_temperature=var_kd_temperature,
                                    var_teacher_epochs=var_teacher_epochs,
                                    var_student_epochs=var_student_epochs,
                                    var_compile=var_compile,
                                    var_repeat=var_repeat, var_task=var_task, var_env=var_env,
                                    save_path=save_path, var_loss_mode="dem")
