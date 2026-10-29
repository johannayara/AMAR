"""
[file]          set_clustering.py
[description]   What the AMAR_WO_RVQ set-prediction head does with a set of CSI windows: which of its
                num_obj_queries slots each sample's occupant is matched to, what each slot predicts,
                which samples the query embeddings group together, and which part of the window each
                of the slot's cross-attention heads reads.

                The head is a set predictor: num_obj_queries learnable queries each emit a class
                distribution (TransformerDecoder.class_embed) and the loss matches them to a sample's
                occupants with the Hungarian algorithm (HungarianMatchingLoss). A sample's occupant
                therefore lands on one specific slot, and the samples that land on the same slot are
                the group the head treats as one thing. The evidence for the grouping is the slot's
                cross-attention, which TransformerDecoderLayer already keeps per head
                (need_weights=True, average_attn_weights=False).

                Two groupings are drawn, because they can disagree and the disagreement is the
                interesting part: the head's own slots (does it spread the activities over several
                slots, or collapse them onto one?) and the clusters of the query embeddings
                (does the representation separate what the slots do not?). Both are labelled with the
                activity they hold, so a cluster reads as "walk, 96% pure, n=58" rather than as an
                anonymous colour.

                The slot index of a *target* carries no meaning -- build_activity_set_labels permutes
                the target query order per sample -- but the model's own slots are shared across
                samples, and those are what the matching recovers.

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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

#
##

## Cap on the points the t-SNE sees: it is O(n^2) in memory and the figure cannot show more anyway.
MAX_TSNE_POINTS = 600
## Cap on the per-cluster attention panels.
MAX_ATTENTION_PANELS = 6
## Cap on the number of embedding clusters, whatever the activity count.
MAX_CLUSTERS = 10
## The attention profile is resampled onto this many relative positions, so the profiles of two models
## whose backbones emit a different number of memory tokens (NTU-Fi HAR 500 packets vs WiMANS 3000)
## are on one x-axis.
NUM_ATTENTION_BINS = 64
## Frames probed to measure how much of a window is leading zero padding.
NUM_PADDING_PROBES = 32


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
    : return: (var_logits (n, Q, C), var_embed (n, Q, d_model), var_attn (n, nhead, Q, num_tokens))
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


def leading_padding_fraction(var_x, var_ids, var_max_probe=NUM_PADDING_PROBES):
    """
    [description]
    : how much of a window is leading zero padding, averaged over a few of the frames. WiMANS captures
      are shorter than the window and are padded at the front (load_data_x's convention), so the head
      attending to the first part of a window may be attending to nothing at all.
    : return: float in [0, 1)
    """
    #
    ##
    var_fracs = []
    for var_id in np.asarray(var_ids)[:var_max_probe]:
        var_nonzero = np.any(np.asarray(var_x[var_id]) != 0, axis=1)
        if var_nonzero.any():
            var_fracs.append(float(np.argmax(var_nonzero)) / float(var_nonzero.shape[0]))
    return float(np.mean(var_fracs)) if var_fracs else 0.0


def cluster_embeddings(var_points, var_num_clusters, var_seed=39):
    """
    [description]
    : KMeans over the query embeddings, i.e. the grouping the representation itself forms, independent
      of the slots the head assigns. Labelling these clusters with their dominant activity is what says
      whether the representation separates the activities.
    : return: (var_labels (n,), var_num_clusters actually used)
    """
    #
    ##
    var_k = int(max(1, min(var_num_clusters, len(var_points))))
    if var_k == 1:
        return np.zeros(len(var_points), dtype=np.int64), 1
    var_kmeans = KMeans(n_clusters=var_k, n_init=10, random_state=var_seed).fit(var_points)
    return var_kmeans.labels_, var_k


def visualize_query_clustering(var_model, var_x, var_y, var_class_names, save_path,
                               var_index=None, var_model_name="model", var_batch_size=32,
                               var_device=None, var_seed=39, var_num_clusters=None):
    """
    [description]
    : draw which samples the set-prediction head groups together and why. Writes
      <save_path>/clustering_<var_model_name>.png and returns the summary.
      Panels: the query embeddings of the matched slots (t-SNE) coloured by true activity; the same
      points coloured by embedding cluster with every cluster annotated with the activity it holds and
      its purity; the cluster x activity and slot x activity contingencies; and one attention panel per
      cluster showing each cross-attention head's profile over the window, with the leading zero
      padding shaded.
    : var_model: a trained AMAR_WO_RVQ (evaluated here, left in eval mode)
    : var_x: (N, time, features) the pool the model consumes
    : var_index: rows of var_x to explain (default all); the runner passes its held-out query frames
    : var_y: (n, num_queries, num_classes) one-hot sets of those frames, no-object class last
    : var_class_names: activity names in class order, without the no-object class
    : var_num_clusters: KMeans k (default: the number of activities present, capped at MAX_CLUSTERS)
    : return: dict, the slot and cluster summaries for the run record
    """
    #
    ##
    var_ids = np.arange(len(var_x)) if var_index is None else np.asarray(var_index)
    var_logits, var_embed, var_attn = forward_with_evidence(
        var_model, var_x, var_index=var_ids, var_batch_size=var_batch_size, var_device=var_device)
    var_slot, var_activity = match_samples_to_slots(var_logits, var_y)
    var_no_object = var_logits.shape[-1] - 1
    var_predicted = np.argmax(var_logits, axis=-1)                      # (n, Q)
    var_used = [int(var_s) for var_s in sorted(set(var_slot.tolist()) - {-1})]
    var_summary = {"slots": {}, "clusters": {}}
    for var_s in var_used:
        var_members = np.where(var_slot == var_s)[0]
        var_counts = np.bincount(var_activity[var_members], minlength=len(var_class_names))
        var_pred = np.bincount(var_predicted[var_members, var_s], minlength=var_no_object + 1)
        var_summary["slots"][str(var_s)] = {
            "samples": int(len(var_members)),
            "top_activity": var_class_names[int(var_counts.argmax())],
            "purity": float(var_counts.max() / max(1, var_counts.sum())),
            "predicted_class": (var_class_names[int(var_pred.argmax())]
                                if var_pred.argmax() < var_no_object else "no-object"),
        }
    #
    ## ---------------------- the grouping the representation itself forms --------------------------
    var_matched = np.where(var_slot >= 0)[0]
    var_points = (var_embed[var_matched, var_slot[var_matched]] if len(var_matched) else
                  np.zeros((0, var_embed.shape[-1]), dtype=np.float32))
    var_acts_matched = var_activity[var_matched]
    var_present = sorted(set(var_acts_matched.tolist()))
    var_k = var_num_clusters or min(len(var_present), MAX_CLUSTERS)
    var_labels, var_k = cluster_embeddings(var_points, var_k, var_seed=var_seed)
    for var_c in range(var_k):
        var_members = var_acts_matched[var_labels == var_c]
        var_counts = np.bincount(var_members, minlength=len(var_class_names))
        var_summary["clusters"][str(var_c)] = {
            "samples": int(len(var_members)),
            "top_activity": var_class_names[int(var_counts.argmax())],
            "purity": float(var_counts.max() / max(1, var_counts.sum())),
            "activities": {var_class_names[var_a]: int(var_n) for var_a, var_n in
                           enumerate(var_counts) if var_n},
        }
    #
    ## --------------------------------------- figure ----------------------------------------------
    var_attn_clusters = list(range(var_k))[:MAX_ATTENTION_PANELS]
    var_num_panels = 4 + len(var_attn_clusters)
    var_ncols = 3
    var_nrows = int(np.ceil(var_num_panels / var_ncols))
    fig = plt.figure(figsize=(5.6 * var_ncols, 4.4 * var_nrows))
    var_grid = fig.add_gridspec(var_nrows, var_ncols)
    var_cells = [var_grid[var_r, var_c] for var_r in range(var_nrows) for var_c in range(var_ncols)]
    var_padding = leading_padding_fraction(var_x, var_ids)
    #
    ## ---- panels 1/2: the samples in the embedding of the slot they were matched to ----------------
    var_ax_activity, var_ax_cluster = fig.add_subplot(var_cells[0]), fig.add_subplot(var_cells[1])
    if len(var_matched) >= 10:
        var_rng = np.random.RandomState(var_seed)
        var_show = (np.arange(len(var_matched)) if len(var_matched) <= MAX_TSNE_POINTS else
                    np.sort(var_rng.choice(len(var_matched), MAX_TSNE_POINTS, replace=False)))
        var_perplexity = max(5.0, min(30.0, (len(var_show) - 1) / 3.0))
        var_tsne = TSNE(n_components=2, perplexity=var_perplexity, random_state=var_seed, init="pca",
                        learning_rate="auto", n_jobs=-1).fit_transform(var_points[var_show])
        ## panel 1: coloured by the activity the frame actually holds, so the visual groups can be read
        var_cmap_act = plt.get_cmap("tab20", max(len(var_present), 1))
        for var_i, var_a in enumerate(var_present):
            var_mask = var_acts_matched[var_show] == var_a
            var_ax_activity.scatter(var_tsne[var_mask, 0], var_tsne[var_mask, 1], s=22, alpha=0.85,
                                    color=var_cmap_act(var_i),
                                    label=f"{var_class_names[var_a]} (n={int(var_mask.sum())})")
        var_ax_activity.legend(fontsize=7, loc="best", ncol=2, markerscale=1.3)
        var_ax_activity.set_title(f"{var_model_name}: query embeddings, coloured by true activity")
        ## panel 2: the same points by embedding cluster, each cluster named by the activity it holds
        var_cmap_cl = plt.get_cmap("tab10", max(var_k, 1))
        for var_c in range(var_k):
            var_mask = var_labels[var_show] == var_c
            var_ax_cluster.scatter(var_tsne[var_mask, 0], var_tsne[var_mask, 1], s=22, alpha=0.85,
                                   color=var_cmap_cl(var_c % 10))
            var_info = var_summary["clusters"][str(var_c)]
            var_ax_cluster.annotate(f"c{var_c}: {var_info['top_activity']}\n"
                                    f"{var_info['purity']:.0%} pure, n={var_info['samples']}",
                                    (float(var_tsne[var_mask, 0].mean()),
                                     float(var_tsne[var_mask, 1].mean())),
                                    fontsize=8, weight="bold", ha="center", va="center",
                                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.8))
        var_ax_cluster.set_title("same points by embedding cluster (KMeans), labelled with what it holds")
    else:
        for var_ax in (var_ax_activity, var_ax_cluster):
            var_ax.text(0.5, 0.5, f"too few matched samples ({len(var_matched)})", ha="center",
                        va="center")
    #
    ## ---- panel 3: which activity each cluster holds ---------------------------------------------
    var_ax_cl_map = fig.add_subplot(var_cells[2])
    var_cl_table = np.zeros((max(var_k, 1), len(var_class_names)))
    for var_c in range(var_k):
        var_cl_table[var_c] = np.bincount(var_acts_matched[var_labels == var_c],
                                          minlength=len(var_class_names))
    var_ax_cl_map.imshow(var_cl_table, aspect="auto", cmap="Blues")
    var_ax_cl_map.set_xticks(range(len(var_class_names)))
    var_ax_cl_map.set_xticklabels(var_class_names, rotation=90, fontsize=7)
    var_ax_cl_map.set_yticks(range(max(var_k, 1)))
    var_ax_cl_map.set_yticklabels([f"c{var_c} ({var_summary['clusters'][str(var_c)]['top_activity']}, "
                                   f"{var_summary['clusters'][str(var_c)]['purity']:.0%})"
                                   for var_c in range(var_k)] or ["-"], fontsize=7)
    for var_r in range(var_k):
        for var_c in range(len(var_class_names)):
            if var_cl_table[var_r, var_c]:
                var_ax_cl_map.text(var_c, var_r, int(var_cl_table[var_r, var_c]), ha="center",
                                   va="center", fontsize=6)
    var_ax_cl_map.set_title("embedding clusters vs true activity (counts)")
    #
    ## ---- panel 4: which activity the head's slots hold -------------------------------------------
    var_ax_map = fig.add_subplot(var_cells[3])
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
    var_ax_map.set_title(f"the head's slots vs true activity "
                         f"({len(var_used)} slot(s) used for {len(var_present)} activities)")
    #
    ## ---- remaining panels: what each cluster's attention heads read ------------------------------
    var_axis_x = np.linspace(0, 1, NUM_ATTENTION_BINS)
    for var_i, var_c in enumerate(var_attn_clusters):
        var_ax = fig.add_subplot(var_cells[4 + var_i])
        var_members = var_matched[var_labels == var_c]
        var_profiles = np.stack([resample_attention(var_attn[var_m, :, var_slot[var_m], :])
                                 for var_m in var_members])
        var_info = var_summary["clusters"][str(var_c)]
        for var_head in range(var_profiles.shape[1]):
            var_mean = var_profiles[:, var_head, :].mean(axis=0)
            var_se = var_profiles[:, var_head, :].std(axis=0) / np.sqrt(len(var_members))
            var_ax.plot(var_axis_x, var_mean, linewidth=1.5, label=f"head {var_head}")
            var_ax.fill_between(var_axis_x, var_mean - var_se, var_mean + var_se, alpha=0.18)
        if var_padding > 0:
            var_ax.axvspan(0, var_padding, color="0.85", zorder=0)
            var_ax.annotate("padding", (var_padding / 2, var_ax.get_ylim()[1] * 0.95), fontsize=7,
                            ha="center", va="top", color="0.35")
        var_ax.set_ylim(bottom=0)
        var_ax.set_title(f"c{var_c} attention over the window — {var_info['top_activity']} "
                         f"({var_info['purity']:.0%} pure, n={var_info['samples']})", fontsize=9)
        var_ax.set_xlabel("relative packet position")
        var_ax.set_ylabel("attention (mean ± SE)")
        if var_i == 0:
            var_ax.legend(fontsize=7)
    fig.suptitle(f"{var_model_name}: head grouping and its evidence — {len(var_used)} slot(s) used, "
                 f"{var_k} embedding cluster(s), {len(var_matched)} matched samples", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"clustering_{var_model_name}.png"), dpi=140)
    plt.close(fig)
    #
    var_summary["slots_used"] = var_used
    var_summary["num_clusters"] = int(var_k)
    var_summary["matched_samples"] = int(len(var_matched))
    var_summary["unmatched_samples"] = int(len(var_slot) - len(var_matched))
    var_summary["leading_padding_fraction"] = var_padding
    return var_summary
