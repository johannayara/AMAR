"""
[file]          set_clustering.py
[description]   What the AMAR_WO_RVQ set-prediction head does with a set of CSI windows: which of its
                num_obj_queries slots each sample's occupant is matched to, what each slot predicts,
                and which part of the window each of that slot's cross-attention heads reads.

                The head is a set predictor: num_obj_queries learnable queries each emit a class
                distribution (TransformerDecoder.class_embed) and the loss matches them to a sample's
                occupants with the Hungarian algorithm (HungarianMatchingLoss). A sample's occupant
                therefore lands on one specific slot, and the samples that land on the same slot are
                the group the head treats as one thing -- that is the "clustering" drawn here. The
                evidence for the grouping is the slot's cross-attention, which TransformerDecoderLayer
                already keeps per head (need_weights=True, average_attn_weights=False), so the figure
                shows for every slot what each head reads over the window.

                The slot index of a *target* carries no meaning -- build_activity_set_labels permutes
                the target query order per sample -- but the model's own slots are shared across
                samples, and those are what the matching recovers. Every sample whose occupant matched
                model slot j is grouped under j.

                This runs on a model that is already in memory (the few-shot runner calls it on its
                frozen teacher and its fine-tuned student); it is not a checkpoint loader.
"""
#
##

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE

#
##

## Cap on the points the t-SNE sees: it is O(n^2) in memory and the figure cannot show more anyway.
MAX_TSNE_POINTS = 500
## Cap on the attention panels: one panel per slot the head actually used.
MAX_ATTENTION_PANELS = 6
## The attention profile is resampled onto this many relative positions, so the profiles of two models
## whose backbones emit a different number of memory tokens (NTU-Fi HAR 500 packets vs WiMANS 3000)
## are on one x-axis.
NUM_ATTENTION_BINS = 64


def match_samples_to_slots(var_logits, var_y):
    """
    [description]
    : the Hungarian matching the loss uses at training time, applied to a batch so that every sample's
      occupant can be traced to the model slot that predicted it.
    : var_logits: (N, num_queries, num_classes) class logits of the model's last decoder layer
    : var_y: (N, num_queries, num_classes) one-hot target sets, no-object class last
    : return: (var_slot (N,) int64 model-slot index per sample, -1 = no occupant matched,
      var_activity (N,) int64 index into the activity names of the matched target, -1 = none)
    """
    #
    ##
    var_num_classes = var_logits.shape[-1]
    var_no_object = var_num_classes - 1
    var_log_prob = torch.log_softmax(torch.from_numpy(var_logits), dim=-1).numpy()
    var_target_class = np.argmax(var_y, axis=-1)
    var_slot = np.full(len(var_logits), -1, dtype=np.int64)
    var_activity = np.full(len(var_logits), -1, dtype=np.int64)
    for var_i in range(len(var_logits)):
        ## cost[i, j] = -log P(model slot i predicts the class of target slot j)
        var_cost = -var_log_prob[var_i][:, var_target_class[var_i]]
        var_rows, var_cols = linear_sum_assignment(var_cost)
        var_real = [(int(var_r), int(var_c)) for var_r, var_c in zip(var_rows, var_cols)
                    if var_target_class[var_i, var_c] != var_no_object]
        if var_real:
            ## a single-occupant frame has one real target and so one group; a multi-occupant frame is
            ## grouped by the slot its first real target was matched to
            var_slot[var_i], var_activity[var_i] = var_real[0][0], var_target_class[var_i, var_real[0][1]]
    return var_slot, var_activity


def forward_with_evidence(var_model, var_x, var_index=None, var_batch_size=32, var_device=None):
    """
    [description]
    : one forward pass that keeps what the head's decision is made of: the last decoder layer's class
      logits, the query embeddings they are read off, and the last layer's per-head cross-attention
      over the memory tokens. The three calls are the ones AMAR_WO_RVQ.forward makes.
    : var_x: (N, time, features) float32 the model consumes; var_index selects rows of it
    : return: (var_logits (n, Q, C), var_embed (n, Q, d_model), var_attn (n, nhead, Q, num_tokens),
      var_lengths (n,) true packet counts if var_x carries padding, else None)
    """
    #
    ##
    var_device = var_device or next(var_model.parameters()).device
    var_ids = np.arange(len(var_x)) if var_index is None else np.asarray(var_index)
    var_logits, var_embed, var_attn = [], [], []
    var_model.eval()
    with torch.no_grad():
        for var_start in range(0, len(var_ids), var_batch_size):
            ## index the pool per batch rather than copying the selected frames up front: the union
            ## pool is several GB and its query slice is 80% of it
            var_batch = var_x[var_ids[var_start:var_start + var_batch_size]]
            var_xb = torch.from_numpy(np.ascontiguousarray(var_batch)).to(var_device)
            if var_model.var_sequence_backbone:
                var_feats = var_model.feature_extractor.forward_sequence(var_xb)
            else:
                var_feats = var_model.feature_extractor(var_xb)
            var_memory = var_model.encoder(var_feats)
            var_pred, var_emb = var_model.decoder(var_memory, return_embedding=True)
            var_logits.append(var_pred[-1].cpu().numpy())
            var_embed.append(var_emb.cpu().numpy())
            var_attn.append(var_model.decoder.decoder_layers[-1].cross_attn_weights.cpu().numpy())
    return np.concatenate(var_logits), np.concatenate(var_embed), np.concatenate(var_attn)


def resample_attention(var_attn, var_num_bins=NUM_ATTENTION_BINS):
    """
    [description]
    : (nhead, num_tokens) attention of one query -> (nhead, var_num_bins), interpolated onto relative
      positions 0..1 and renormalized, so profiles from backbones with different token counts compare.
    """
    #
    ##
    var_src = np.linspace(0.0, 1.0, var_attn.shape[-1])
    var_dst = np.linspace(0.0, 1.0, var_num_bins)
    var_out = np.stack([np.interp(var_dst, var_src, var_row) for var_row in var_attn])
    return var_out / (var_out.sum(axis=-1, keepdims=True) + 1e-12)


def visualize_query_clustering(var_model, var_x, var_y, var_class_names, save_path,
                               var_index=None, var_model_name="model", var_batch_size=32,
                               var_device=None, var_seed=39):
    """
    [description]
    : draw which samples the set-prediction head groups together and why. Writes
      <save_path>/clustering_<var_model_name>.png and returns the per-slot summary.
      Panels: the query embeddings of the matched slots (t-SNE) coloured by slot, the same points
      coloured by the true activity, the slot x activity contingency, and one attention panel per used
      slot showing each cross-attention head's profile over the window.
    : var_model: a trained AMAR_WO_RVQ (evaluated here, left in eval mode)
    : var_x: (N, time, features) the pool the model consumes
    : var_index: rows of var_x to explain (default all); the runner passes its held-out query frames
    : var_y: (n, num_queries, num_classes) one-hot sets of those frames, no-object class last
    : var_class_names: activity names in class order, without the no-object class
    : return: dict, the per-slot summary for the run record
    """
    #
    ##
    var_logits, var_embed, var_attn = forward_with_evidence(
        var_model, var_x, var_index=var_index, var_batch_size=var_batch_size, var_device=var_device)
    var_slot, var_activity = match_samples_to_slots(var_logits, var_y)
    var_no_object = var_logits.shape[-1] - 1
    ## what each slot says it sees, over the samples it was matched to
    var_predicted = np.argmax(var_logits, axis=-1)                      # (n, Q)
    var_used = [int(var_s) for var_s in sorted(set(var_slot.tolist()) - {-1})]
    var_summary = {}
    for var_s in var_used:
        var_members = np.where(var_slot == var_s)[0]
        var_acts = var_activity[var_members]
        var_counts = np.bincount(var_acts, minlength=len(var_class_names))
        var_pred = np.bincount(var_predicted[var_members, var_s], minlength=var_no_object + 1)
        var_summary[str(var_s)] = {
            "samples": int(len(var_members)),
            "top_activity": var_class_names[int(var_counts.argmax())],
            "purity": float(var_counts.max() / max(1, var_counts.sum())),
            "predicted_class": (var_class_names[int(var_pred.argmax())] if var_pred.argmax() < var_no_object
                                else "no-object"),
        }
    #
    ## --------------------------------------- figure ----------------------------------------------
    var_num_panels = 3 + len(var_used[:MAX_ATTENTION_PANELS])
    var_ncols = 3
    var_nrows = int(np.ceil(var_num_panels / var_ncols))
    fig = plt.figure(figsize=(5.5 * var_ncols, 4.2 * var_nrows))
    var_grid = fig.add_gridspec(var_nrows, var_ncols)
    var_cells = [var_grid[var_r, var_c] for var_r in range(var_nrows) for var_c in range(var_ncols)]
    #
    ## ---- panel 1/2: the samples in the embedding of the slot they were matched to ----------------
    var_ax_slot, var_ax_activity = fig.add_subplot(var_cells[0]), fig.add_subplot(var_cells[1])
    var_matched = np.where(var_slot >= 0)[0]
    if len(var_matched) >= 10:
        var_rng = np.random.RandomState(var_seed)
        var_show = (var_matched if len(var_matched) <= MAX_TSNE_POINTS else
                    np.sort(var_rng.choice(var_matched, MAX_TSNE_POINTS, replace=False)))
        var_points = var_embed[var_show, var_slot[var_show]]
        var_perplexity = max(5.0, min(30.0, (len(var_show) - 1) / 3.0))
        var_tsne = TSNE(n_components=2, perplexity=var_perplexity, random_state=var_seed, init="pca",
                        learning_rate="auto", n_jobs=-1).fit_transform(var_points)
        var_colors = plt.get_cmap("tab10", max(len(var_used), 1))
        var_lookup = {var_s: var_i for var_i, var_s in enumerate(var_used)}
        for var_s in var_used:
            var_mask = var_slot[var_show] == var_s
            var_label = f"slot {var_s} ({var_summary[str(var_s)]['top_activity']}, " \
                        f"n={var_summary[str(var_s)]['samples']})"
            var_ax_slot.scatter(var_tsne[var_mask, 0], var_tsne[var_mask, 1], s=22, alpha=0.85,
                                color=var_colors(var_lookup[var_s]), label=var_label)
            var_ax_activity.scatter(var_tsne[var_mask, 0], var_tsne[var_mask, 1], s=22, alpha=0.85,
                                    color=var_colors(var_lookup[var_s]))
        var_ax_slot.legend(fontsize=6, loc="best", markerscale=1.2)
        var_ax_slot.set_title(f"{var_model_name}: samples by matched slot (t-SNE of query embeddings)")
        ## the same points, with the slot's activity written on its centroid, to read the mapping off
        for var_s in var_used:
            var_mask = var_slot[var_show] == var_s
            var_ax_activity.annotate(var_summary[str(var_s)]["top_activity"],
                                     (float(var_tsne[var_mask, 0].mean()), float(var_tsne[var_mask, 1].mean())),
                                     fontsize=7, weight="bold", ha="center")
        var_ax_activity.set_title("same points, labelled with the slot's dominant activity")
    else:
        for var_ax in (var_ax_slot, var_ax_activity):
            var_ax.text(0.5, 0.5, f"too few matched samples ({len(var_matched)})", ha="center",
                        va="center")
    #
    ## ---- panel 3: which activity each slot is used for ------------------------------------------
    var_ax_map = fig.add_subplot(var_cells[2])
    var_table = np.zeros((max(len(var_used), 1), len(var_class_names)))
    for var_i, var_s in enumerate(var_used):
        var_table[var_i] = np.bincount(var_activity[var_slot == var_s], minlength=len(var_class_names))
    var_ax_map.imshow(var_table, aspect="auto", cmap="Blues")
    var_ax_map.set_xticks(range(len(var_class_names)))
    var_ax_map.set_xticklabels(var_class_names, rotation=90, fontsize=7)
    var_ax_map.set_yticks(range(max(len(var_used), 1)))
    var_ax_map.set_yticklabels([f"slot {var_s}" for var_s in var_used] or ["-"], fontsize=7)
    for var_r in range(len(var_used)):
        for var_c in range(len(var_class_names)):
            if var_table[var_r, var_c]:
                var_ax_map.text(var_c, var_r, int(var_table[var_r, var_c]), ha="center", va="center",
                                fontsize=6)
    var_ax_map.set_title("samples matched to each slot, by true activity")
    #
    ## ---- remaining panels: what each slot's attention heads read --------------------------------
    for var_i, var_s in enumerate(var_used[:MAX_ATTENTION_PANELS]):
        var_ax = fig.add_subplot(var_cells[3 + var_i])
        var_members = np.where(var_slot == var_s)[0]
        ## (members, nhead, tokens) -> (members, nhead, bins)
        var_profiles = np.stack([resample_attention(var_attn[var_m, :, var_s, :])
                                 for var_m in var_members])
        for var_head in range(var_profiles.shape[1]):
            var_ax.plot(np.linspace(0, 1, var_profiles.shape[-1]),
                        var_profiles[:, var_head, :].mean(axis=0), linewidth=1.4,
                        label=f"head {var_head}")
            var_ax.fill_between(np.linspace(0, 1, var_profiles.shape[-1]),
                                var_profiles[:, var_head, :].mean(axis=0)
                                - var_profiles[:, var_head, :].std(axis=0),
                                var_profiles[:, var_head, :].mean(axis=0)
                                + var_profiles[:, var_head, :].std(axis=0), alpha=0.12)
        var_ax.set_ylim(bottom=0)
        var_ax.set_title(f"slot {var_s} attention over the window "
                         f"({var_summary[str(var_s)]['top_activity']}, n={len(var_members)})",
                         fontsize=9)
        var_ax.set_xlabel("relative packet position")
        var_ax.set_ylabel("attention (per head)")
        if var_i == 0:
            var_ax.legend(fontsize=7)
    fig.suptitle(f"{var_model_name}: set-prediction head grouping and its evidence "
                 f"({len(var_used)} slots used, {len(var_matched)} matched samples)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"clustering_{var_model_name}.png"), dpi=140)
    plt.close(fig)
    #
    return {
        "slots_used": var_used,
        "matched_samples": int(len(var_matched)),
        "unmatched_samples": int(len(var_slot) - len(var_matched)),
        "per_slot": var_summary,
    }
