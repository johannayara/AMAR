"""
[file]          that_variance.py
[description]   THAT-style CSI encoder extended for localization ("THATV").

                It keeps the two-stream convolution-augmented transformer design of THAT
                (src/models/bce_that.py) and adds three localization-specific branches:

                1. Temporal statistics. THAT never sees an explicit motion cue: a stationary room and
                   a moving one can produce similar per-token features. THIS backbone computes
                   per-feature temporal statistics of the raw window (mean, std, and first-difference
                   energy, i.e. how much the CSI amplitude varies over time) and adds the projected
                   summary to every time token *after* the left encoder's LayerNorm, so the injected
                   shift reaches the convs instead of being re-normalised away. This is the "CSI
                   variance over time" feature.
                2. Tokens are kept, not summed. THAT collapses the whole window with a global sum, so
                   a head can only ever emit one vector per window and all "where/when" detail must
                   survive that sum. THIS backbone exposes the left stream's per-time-step tokens
                   through forward_sequence(), so sequence heads (AMAR) can attend over time.
                3. Attention pooling as a residual. The single-vector output (density_map) keeps
                   THAT's unweighted sum of the left convs as its base and ADDS a learned
                   softmax-weighted average of the same tokens.

                The three branches are zero-initialised (reset_residual_branches), so a freshly built
                THATV reproduces THAT's single-vector aggregation exactly (the added terms are 0) and
                the optimizer grows them from there. This matters because an earlier revision of this
                file *replaced* THAT's sum with attention pooling and a final LayerNorm, which removed
                the total-mass cue the density head needs to count people: on the classroom run it
                dropped Exact-count accuracy from 0.62 to 0.38 and Count MAE from 0.40 to 0.78
                (output/density_map/{default,thatv}/classroom/run_20261012_164132.log).

                Each of the three features can be switched off independently for ablation. Set
                nn.backbone to "thatv-nostats", "thatv-notokens" or "thatv-nopool" (or a comma list
                such as "thatv-nostats,nopool") to disable the named features; "thatv" keeps all three
                and "default" selects plain THAT. See parse_backbone. The "tokens" ablation restores
                THAT's valid-conv geometry (the time axis shrinks and the two conv branches no longer
                align), so it is only usable by single-vector heads: forward_sequence needs the
                length-preserving convs and raises otherwise.

                Two entry points, matching the two consumers:
                  - forward(x)          -> (B, D)          for single-vector heads (DensityMapNet)
                  - forward_sequence(x) -> (B, N, D)       for sequence heads (AMAR_WO_RVQ)

                Select it with `--hp nn.backbone=thatv` (see configs/preset.py). The head width D is
                whatever the consumer asks for (embedding_dim for density_map, d_embedding for AMAR).
"""
#
##

import torch
import torch.nn as nn

from src.models.bce_that import Encoder, Gaussian_Position, SamePadConv1d


def _pick_heads(var_dim, var_max_heads=10):
    """
    [description]
    : largest attention-head count up to var_max_heads that divides var_dim. THAT hardcodes 10 heads,
      which silently requires both the feature count and the pooled time length to be multiples of 10;
      this keeps the model runnable for any window length.
    """
    for var_heads in range(min(var_max_heads, var_dim), 0, -1):
        if var_dim % var_heads == 0:
            return var_heads
    return 1


class THATV(nn.Module):
    """
    [description]
    : CSI encoder for localization. Left stream ("time-over-channel") produces one token per pooled
      time step and keeps them; right stream ("channel-over-time") is pooled as in THAT. The
      single-vector output is THAT's sum aggregation plus an attention-pooled residual, so the
      localization branches add capacity on top of the baseline rather than replacing it.
    """

    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape,
                 var_use_stats=True,
                 var_use_tokens=True,
                 var_use_pool=True):
        #
        ##
        super().__init__()
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        var_dim_output = var_y_shape[-1]

        var_num_gaussian = 10
        var_time_pool = 20
        var_left_blocks = 4
        var_dropout = 0.5

        var_pooled_time = max(1, var_dim_time // var_time_pool)
        self.var_dim_feature = var_dim_feature
        self.var_pooled_time = var_pooled_time
        #
        ## ablation switches for the three localization features (see module docstring / parse_backbone)
        self.var_use_stats = var_use_stats
        self.var_use_tokens = var_use_tokens
        self.var_use_pool = var_use_pool
        #
        ## ---- temporal statistics branch: raw per-feature mean / std / |first difference| ----
        ## this is where "CSI variance over time" enters the features
        if var_use_stats:
            self.layer_stats = nn.Sequential(
                nn.Linear(var_dim_feature * 3, var_dim_feature),
                nn.LeakyReLU(),
            )
        #
        ## ---- left stream: time-over-channel (tokens kept) ----
        self.layer_left_pooling = nn.AvgPool1d(var_time_pool, var_time_pool)
        self.layer_left_gaussian = Gaussian_Position(var_dim_feature, var_pooled_time, var_num_gaussian)
        self.layer_left_encoder = nn.ModuleList([
            Encoder(var_dim_feature, _pick_heads(var_dim_feature), [1, 3, 5])
            for _ in range(var_left_blocks)])
        self.layer_left_norm = nn.LayerNorm(var_dim_feature, eps=1e-6)
        ## length-preserving so the two branches keep the same number of tokens (THAT could use plain
        ## Conv1d because it summed the time axis away; we keep it). Ablating "tokens" restores THAT's
        ## valid-conv geometry: the time axis shrinks and the two branches no longer align.
        if var_use_tokens:
            self.layer_left_cnn_0 = SamePadConv1d(var_dim_feature, 128, 8)
            self.layer_left_cnn_1 = SamePadConv1d(var_dim_feature, 128, 16)
        else:
            self.layer_left_cnn_0 = nn.Conv1d(var_dim_feature, 128, 8)
            self.layer_left_cnn_1 = nn.Conv1d(var_dim_feature, 128, 16)
        self.layer_left_dropout = nn.Dropout(var_dropout)
        #
        ## ---- right stream: channel-over-time (pooled) ----
        self.layer_right_pooling = nn.AvgPool1d(var_time_pool, var_time_pool)
        self.layer_right_encoder = nn.ModuleList([
            Encoder(var_pooled_time, _pick_heads(var_pooled_time), [1, 2, 3])])
        self.layer_right_norm = nn.LayerNorm(var_pooled_time, eps=1e-6)
        self.layer_right_cnn_0 = nn.Conv1d(var_pooled_time, 16, 2)
        self.layer_right_cnn_1 = nn.Conv1d(var_pooled_time, 16, 4)
        self.layer_right_dropout = nn.Dropout(var_dropout)
        #
        self.layer_leakyrelu = nn.LeakyReLU()
        #
        ## ---- heads ----
        ## the single-vector head is THAT's Linear(256 + 32 -> D); the attention-pooled residual is
        ## added on top and starts at zero
        if var_use_pool or var_use_tokens:
            self.layer_token_proj = nn.Linear(256, var_dim_output)
        if var_use_pool:
            self.layer_pool_score = nn.Linear(var_dim_output, 1)
            self.layer_pool_proj = nn.Linear(256, var_dim_output)
        self.layer_output = nn.Linear(256 + 32, var_dim_output)
        #
        self.reset_residual_branches()

    #
    ##
    def reset_residual_branches(self):
        """
        [description]
        : zero the three localization-specific branches (temporal statistics, attention scores,
          attention residual). With them at zero, forward() is exactly THAT's single-vector encoder:
          sum the left convs over time, concat the right pooled vector, one Linear. Training then
          starts from the baseline solution and adds the localization capacity only as far as it
          lowers the loss, instead of starting from a randomly-initialised different aggregation.
        """
        if self.var_use_stats:
            nn.init.zeros_(self.layer_stats[0].weight)
            nn.init.zeros_(self.layer_stats[0].bias)
        if self.var_use_pool:
            nn.init.zeros_(self.layer_pool_score.weight)
            nn.init.zeros_(self.layer_pool_score.bias)
            nn.init.zeros_(self.layer_pool_proj.weight)
            nn.init.zeros_(self.layer_pool_proj.bias)

    #
    ##
    def temporal_statistics(self, var_x):
        """
        [description]
        : per-feature temporal statistics of the raw window, projected to a feature-sized summary:
          mean (static level), std (variance over time) and the mean absolute first difference
          (short-term motion energy). Shape (B, F). Injecting these gives the encoder an explicit
          "how much did the channel vary" cue that attention over a summed window cannot recover.
        """
        var_mean = var_x.mean(dim=1)
        var_std = var_x.std(dim=1, unbiased=False)
        if var_x.shape[1] > 1:
            var_diff = var_x.diff(dim=1).abs().mean(dim=1)
        else:
            var_diff = torch.zeros_like(var_mean)
        return self.layer_stats(torch.cat([var_mean, var_std, var_diff], dim=-1))

    #
    ##
    def _left_conv(self, var_input):
        """
        [return]
        : (var_left_0, var_left_1): (B, 128, N) each, the two length-preserving conv branches over
          the pooled time tokens, before any time aggregation.
        """
        var_stats = self.temporal_statistics(var_input) if self.var_use_stats else None
        #
        var_left = self.layer_left_pooling(var_input.permute(0, 2, 1)).permute(0, 2, 1)
        var_left = self.layer_left_gaussian(var_left)
        for var_layer in self.layer_left_encoder:
            var_left = var_layer(var_left)
        #
        ## inject the temporal statistics AFTER the encoder's LayerNorm: added before it, the
        ## per-token normalisation would strip most of the shift again
        var_left = self.layer_left_norm(var_left)
        if var_stats is not None:
            var_left = var_left + var_stats.unsqueeze(1)
        var_left = var_left.permute(0, 2, 1)
        var_left_0 = self.layer_leakyrelu(self.layer_left_cnn_0(var_left))
        var_left_1 = self.layer_leakyrelu(self.layer_left_cnn_1(var_left))
        #
        return var_left_0, var_left_1

    #
    ##
    def _right_pooled(self, var_input):
        """
        [return]
        : (B, 32): THAT's right stream, pooled over the channel axis and summed over time.
        """
        var_right = self.layer_right_pooling(var_input.permute(0, 2, 1))
        for var_layer in self.layer_right_encoder:
            var_right = var_layer(var_right)
        var_right = self.layer_right_norm(var_right).permute(0, 2, 1)
        var_right_0 = self.layer_leakyrelu(self.layer_right_cnn_0(var_right))
        var_right_1 = self.layer_leakyrelu(self.layer_right_cnn_1(var_right))
        var_right = torch.cat([var_right_0.sum(-1), var_right_1.sum(-1)], dim=-1)
        #
        return self.layer_right_dropout(var_right)

    #
    ##
    def forward_sequence(self, var_input):
        """
        [return]
        : tokens: (B, N, D) with N = pooled time steps, for sequence heads (AMAR_WO_RVQ).
        """
        if not self.var_use_tokens:
            raise RuntimeError(
                "THATV.forward_sequence requires the length-preserving convs (feature 'tokens'); "
                "backbone 'thatv-notokens' only supports the single-vector forward().")
        var_left_0, var_left_1 = self._left_conv(var_input)
        var_left = torch.cat([var_left_0, var_left_1], dim=1)
        var_left = self.layer_left_dropout(var_left)
        #
        return self.layer_token_proj(var_left.permute(0, 2, 1))

    #
    ##
    def forward(self, var_input):
        """
        [return]
        : embedding: (B, D) single vector per window, for single-vector heads (DensityMapNet).
        """
        var_left_0, var_left_1 = self._left_conv(var_input)
        #
        ## THAT base: sum each left conv over time, concat the right pooled vector, one Linear. The
        ## sum (not a mean) keeps the total-mass cue the density head reads its count from.
        var_left = torch.cat([var_left_0.sum(-1), var_left_1.sum(-1)], dim=-1)
        var_left = self.layer_left_dropout(var_left)
        var_right = self._right_pooled(var_input)
        var_output = self.layer_output(torch.cat([var_left, var_right], dim=-1))
        #
        if not self.var_use_pool:
            return var_output
        #
        ## localization residual: softmax-weighted average of the same left tokens. Scores and
        ## projection are zero-init, so this term is 0 at build time and only learns if it helps.
        ## Crop to the common length so the two branches form one token tensor: a no-op when the convs
        ## are length-preserving, and the alignment that the 'notokens' valid convs need otherwise.
        var_len = min(var_left_0.shape[-1], var_left_1.shape[-1])
        var_tokens = torch.cat([var_left_0[..., :var_len], var_left_1[..., :var_len]], dim=1)
        var_scores = self.layer_pool_score(self.layer_token_proj(var_tokens.permute(0, 2, 1)))
        var_weights = torch.softmax(var_scores.squeeze(-1), dim=1)
        var_pooled = (var_tokens * var_weights.unsqueeze(1)).sum(dim=-1)
        #
        return var_output + self.layer_pool_proj(var_pooled)


#
##
## ------------------------------------------------------------------------------------------ ##
## ------------------------------ backbone string -> THATV config --------------------------- ##
## ------------------------------------------------------------------------------------------ ##
#
## the three localization features THATV adds on top of THAT, addressed by their ablation names
THATV_FEATURES = ("stats", "tokens", "pool")
THATV_PREFIX = "thatv-"


def parse_backbone(var_backbone):
    """
    [return]
    : (var_is_thatv, var_disabled): whether a preset `nn.backbone` string selects THATV, and the set of
      THATV features it switches off.
        "default"                 -> (False, set())
        "thatv"                   -> (True,  set())
        "thatv-nostats"           -> (True,  {"stats"})
        "thatv-nostats,nopool"    -> (True,  {"stats", "pool"})
    : This is the single place that decodes `nn.backbone`, so DensityMapNet and AMAR_WO_RVQ agree on
      which backbone they built.
    """
    var_backbone = var_backbone or "default"
    if var_backbone == "thatv":
        return True, set()
    if not var_backbone.startswith(THATV_PREFIX):
        return False, set()
    var_disabled = set()
    for var_token in var_backbone[len(THATV_PREFIX):].split(","):
        var_token = var_token.strip()
        if not var_token:
            continue
        if not var_token.startswith("no") or var_token[2:] not in THATV_FEATURES:
            raise ValueError(
                f"nn.backbone={var_backbone!r}: expected 'no' + one of {THATV_FEATURES}, "
                f"got {var_token!r}")
        var_disabled.add(var_token[2:])
    return True, var_disabled


def build_thatv(var_x_shape, var_y_shape, var_backbone):
    """
    [return]
    : a THATV with the features named in var_backbone disabled, e.g. "thatv-nostats".
    """
    var_is_thatv, var_disabled = parse_backbone(var_backbone)
    if not var_is_thatv:
        raise ValueError(f"nn.backbone={var_backbone!r} does not select THATV")
    return THATV(var_x_shape, var_y_shape,
                 var_use_stats="stats" not in var_disabled,
                 var_use_tokens="tokens" not in var_disabled,
                 var_use_pool="pool" not in var_disabled)
