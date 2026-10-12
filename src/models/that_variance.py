"""
[file]          that_variance.py
[description]   THAT-style CSI encoder extended for localization ("THATV").

                It keeps the two-stream convolution-augmented transformer design of THAT
                (src/models/bce_that.py) but changes the three things that hurt localization:

                1. Temporal statistics. THAT never sees an explicit motion cue: a stationary room and
                   a moving one can produce similar per-token features. THIS backbone computes
                   per-feature temporal statistics of the raw window (mean, std, and first-difference
                   energy, i.e. how much the CSI amplitude varies over time) and injects the projected
                   summary into every time token. This is the "CSI variance over time" feature.
                2. Tokens are kept, not summed. THAT collapses the whole window with a global sum, so
                   a head can only ever emit one vector per window and all "where/when" detail must
                   survive that sum. THIS backbone exposes the left stream's per-time-step tokens
                   through forward_sequence(), so sequence heads (AMAR) can attend over time.
                3. Attention pooling instead of a sum. The single-vector output (density_map) is a
                   learned softmax-weighted average of the tokens plus the right stream's pooled
                   features, not an unweighted sum.

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
      time step and keeps them; right stream ("channel-over-time") is pooled as in THAT. Both receive
      the temporal-statistics summary through the left tokens.
    """

    #
    ##
    def __init__(self,
                 var_x_shape,
                 var_y_shape,
                 var_num_gaussian=10,
                 var_time_pool=20,
                 var_left_blocks=4,
                 var_dropout=0.5):
        #
        ##
        super().__init__()
        var_dim_feature = var_x_shape[-1]
        var_dim_time = var_x_shape[-2]
        var_dim_output = var_y_shape[-1]
        var_pooled_time = max(1, var_dim_time // var_time_pool)
        self.var_dim_feature = var_dim_feature
        self.var_pooled_time = var_pooled_time
        #
        ## ---- temporal statistics branch: raw per-feature mean / std / |first difference| ----
        ## this is where "CSI variance over time" enters the features
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
        ## Conv1d because it summed the time axis away; we keep it)
        self.layer_left_cnn_0 = SamePadConv1d(var_dim_feature, 128, 8)
        self.layer_left_cnn_1 = SamePadConv1d(var_dim_feature, 128, 16)
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
        ## ---- heads: left tokens and right pooled -> the requested embedding width ----
        self.layer_token_proj = nn.Linear(256, var_dim_output)
        self.layer_right_proj = nn.Linear(32, var_dim_output)
        self.layer_pool_score = nn.Linear(var_dim_output, 1)
        self.layer_out_norm = nn.LayerNorm(var_dim_output, eps=1e-6)

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
    def forward_sequence(self, var_input):
        """
        [return]
        : tokens: (B, N, D) with N = pooled time steps, for sequence heads (AMAR_WO_RVQ).
        """
        var_stats = self.temporal_statistics(var_input)
        #
        ## left stream, keeping the time axis
        var_left = self.layer_left_pooling(var_input.permute(0, 2, 1)).permute(0, 2, 1)
        var_left = self.layer_left_gaussian(var_left) + var_stats.unsqueeze(1)
        for var_layer in self.layer_left_encoder:
            var_left = var_layer(var_left)
        var_left = self.layer_left_norm(var_left).permute(0, 2, 1)
        var_left_0 = self.layer_leakyrelu(self.layer_left_cnn_0(var_left))
        var_left_1 = self.layer_leakyrelu(self.layer_left_cnn_1(var_left))
        var_left = self.layer_left_dropout(torch.cat([var_left_0, var_left_1], dim=1))
        #
        return self.layer_token_proj(var_left.permute(0, 2, 1))

    #
    ##
    def forward(self, var_input):
        """
        [return]
        : embedding: (B, D) single vector per window, for single-vector heads (DensityMapNet).
        """
        var_tokens = self.forward_sequence(var_input)
        #
        ## right stream, pooled over the channel axis as in THAT
        var_right = self.layer_right_pooling(var_input.permute(0, 2, 1))
        for var_layer in self.layer_right_encoder:
            var_right = var_layer(var_right)
        var_right = self.layer_right_norm(var_right).permute(0, 2, 1)
        var_right_0 = self.layer_leakyrelu(self.layer_right_cnn_0(var_right))
        var_right_1 = self.layer_leakyrelu(self.layer_right_cnn_1(var_right))
        var_right = self.layer_right_dropout(torch.cat([var_right_0.sum(-1), var_right_1.sum(-1)], dim=-1))
        #
        ## learned attention pooling over the left tokens + the right pooled features
        var_weights = torch.softmax(self.layer_pool_score(var_tokens).squeeze(-1), dim=1)
        var_pooled = (var_tokens * var_weights.unsqueeze(-1)).sum(dim=1)
        return self.layer_out_norm(var_pooled + self.layer_right_proj(var_right))
