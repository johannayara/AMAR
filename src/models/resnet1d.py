"""
[file]          resnet1d.py
[description]   ResNet1D for CSI windows, after the WiMANS / ResNet1D design, plus a
                ResNet1DLocalizer that is a drop-in for RoomLocalizer.

                The window is (batch, time, features): the convolution runs along time and the feature
                axis (antennas x subcarriers) is the channel axis, so the paper's "C1D 7 x 1 / 3 x 1 /
                1 x 1" kernels are Conv1d kernels along time:
                  - stem: C1D 7 -> 128, batch norm, ReLU, max pool
                  - RB1..RB4, each one basic residual block:
                        f(x)  = C1D 3 -> 128, BN, ReLU, C1D 3 -> 128, BN
                        skip  = C1D 1 -> 128, BN        (matches the channel count so f(x) + x works)
                        out   = ReLU(f(x) + skip(x))
                  - global average pool -> linear head

                This is the ResNet1D-[1,1,1,1] of the paper (4 residual blocks, one basic block each).
                Its parallel activity head and its softmax-over-locations head are not reproduced: the
                localization runner regresses one position in a per-run shared [0, 1]^2 frame so a model
                can be scored on rooms it never trained on (src/models/room_localization.py), which a
                fixed per-room location vocabulary cannot express. The head here is its own small MLP to
                (x, y), paired with the ResNet1D backbone; the two localizers only share the
                (batch, time, features) -> (batch, 2) contract, not the head.

                Purpose: a small, well-known CSI architecture to test whether the THAT backbone's
                collapse to the mean position (see scripts/test_scripts/diagnose_localizer_fit.py) is a
                property of the backbone or of the task.
"""
#
##

import torch.nn as nn

#
##


class BasicBlock1D(nn.Module):
    """
    [description]
    : the basic residual block of ResNet1D: two 3-tap convolutions with batch norm (ReLU after the
      first) plus a 1-tap batch-normed shortcut, out = ReLU(f(x) + shortcut(x)).
    : var_in_channels / var_out_channels: the shortcut is a 1-tap convolution that maps the input to
      var_out_channels, which is what makes the element-wise addition legal.
    """
    #
    ##
    def __init__(self, var_in_channels, var_out_channels, var_kernel_size=3):
        #
        super().__init__()
        var_pad = var_kernel_size // 2
        self.layer_conv_1 = nn.Conv1d(var_in_channels, var_out_channels, var_kernel_size,
                                      padding=var_pad, bias=False)
        self.layer_norm_1 = nn.BatchNorm1d(var_out_channels)
        self.layer_conv_2 = nn.Conv1d(var_out_channels, var_out_channels, var_kernel_size,
                                      padding=var_pad, bias=False)
        self.layer_norm_2 = nn.BatchNorm1d(var_out_channels)
        self.layer_shortcut = nn.Conv1d(var_in_channels, var_out_channels, 1, bias=False)
        self.layer_shortcut_norm = nn.BatchNorm1d(var_out_channels)
        self.layer_act = nn.ReLU(inplace=True)

    #
    ##
    def forward(self, var_input):
        #
        var_main = self.layer_act(self.layer_norm_1(self.layer_conv_1(var_input)))
        var_main = self.layer_norm_2(self.layer_conv_2(var_main))
        var_skip = self.layer_shortcut_norm(self.layer_shortcut(var_input))
        return self.layer_act(var_main + var_skip)


class ResNet1D(nn.Module):
    """
    [description]
    : the ResNet1D backbone. Input is a (batch, time, features) CSI window; it is transposed internally
      to the (batch, channels, length) layout Conv1d expects.
    : var_num_features: feature count of the CSI window (the Conv1d channel count)
    : var_num_blocks: residual blocks per stage; (1, 1, 1, 1) is the paper's ResNet1D-[1,1,1,1]
    : var_channels / var_stem_channels / var_stem_kernel: the paper's 128 channels and 7-tap stem
    : var_embedding_dim: width of the pooled feature the head consumes
    """
    #
    ##
    def __init__(self, var_num_features, var_num_blocks=(1, 1, 1, 1), var_channels=128,
                 var_stem_channels=128, var_stem_kernel=7, var_embedding_dim=128):
        #
        super().__init__()
        var_stem_pad = var_stem_kernel // 2
        self.layer_stem = nn.Conv1d(var_num_features, var_stem_channels, var_stem_kernel,
                                    padding=var_stem_pad, bias=False)
        self.layer_stem_norm = nn.BatchNorm1d(var_stem_channels)
        self.layer_pool = nn.MaxPool1d(2)
        self.layer_act = nn.ReLU(inplace=True)
        #
        var_blocks, var_in = [], var_stem_channels
        for var_count in var_num_blocks:
            for _ in range(var_count):
                var_blocks.append(BasicBlock1D(var_in, var_channels))
                var_in = var_channels
        self.layer_blocks = nn.Sequential(*var_blocks)
        self.layer_pool_out = nn.AdaptiveAvgPool1d(1)
        self.layer_proj = (nn.Linear(var_in, var_embedding_dim) if var_in != var_embedding_dim
                           else nn.Identity())

    #
    ##
    def forward(self, var_input):
        #
        var_x = var_input.transpose(1, 2)                       # (batch, features, time)
        var_x = self.layer_act(self.layer_stem_norm(self.layer_stem(var_x)))
        var_x = self.layer_pool(var_x)
        var_x = self.layer_blocks(var_x)
        var_x = self.layer_pool_out(var_x).flatten(1)           # (batch, channels)
        return self.layer_proj(var_x)


class ResNet1DLocalizer(nn.Module):
    """
    [description]
    : RoomLocalizer-shaped wrapper: the same (batch, time, features) -> (batch, 2) contract, so
      src.models.room_localization.run_few_shot_localization can build either localizer through its
      var_model_factory without touching the training or the metric code. Its own head is paired with
      the ResNet1D backbone.
    : var_x_shape: (time, features) of one CSI window
    """
    #
    ##
    def __init__(self, var_x_shape, var_num_blocks=(1, 1, 1, 1), var_channels=128,
                 var_embedding_dim=128, var_hidden_dim=128, var_dropout=0.2):
        #
        super().__init__()
        self.backbone = ResNet1D(var_num_features=var_x_shape[-1], var_num_blocks=var_num_blocks,
                                 var_channels=var_channels, var_embedding_dim=var_embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(var_embedding_dim, var_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(var_dropout),
            nn.Linear(var_hidden_dim, 2))

    #
    ##
    def forward(self, var_input):
        #
        return self.head(self.backbone(var_input))
