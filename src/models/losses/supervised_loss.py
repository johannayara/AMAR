
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatchingLoss(nn.Module):
    def __init__(self, cost_class_weight, aux_loss_weight, label_smoothing, class_imbalance_weight, num_classes = 10):
        super().__init__()
        self.cost_class = cost_class_weight
        self.aux_loss_weight = aux_loss_weight

        weights = torch.ones(num_classes)
        weights[-1] = class_imbalance_weight
        weights = weights * (len(weights) / weights.sum())

        # Get the device based on the hierarchy: CUDA, MPS, CPU
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            target_device = torch.device("mps")
        else:
            target_device = torch.device("cpu")
            
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weights.to(target_device),
            label_smoothing=label_smoothing
        )

    @torch.no_grad()
    def Hungarian_matching(self, outputs, targets):
        """
        Performs the matching between predictions and ground truth
        Args:
            outputs: Tensor of shape (batch_size, num_queries, num_classes)
            targets: Tensor of shape (batch_size, num_queries, num_classes)
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = outputs.shape[:2]

        # Compute classification cost matrix
        out_prob = outputs.softmax(-1)  # [batch_size, num_queries, num_classes]
        tgt_ids = targets.argmax(-1)  # [batch_size, num_queries]

        indices = []
        # Process each batch independently
        for b in range(bs):
            # Compute cost matrix for current
            # Create cost matrix showing how well each prediction matches each target
            cost_matrix = -out_prob[b][:, tgt_ids[b]]  # [num_queries, num_queries]
            cost_matrix = self.cost_class * cost_matrix.cpu().numpy()

            # Run Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Convert to tensors and move to correct device
            row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=outputs.device) # Which queries to use
            col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=outputs.device) # Which targets they match to
            indices.append((row_ind, col_ind))
            """
            tgt_ids[b]
            Out[1]: tensor([1, 1, 1, 1, 9], device='cuda:0')
            Example how it works:
            [[-0.1645718 , -0.1645718 , -0.1645718 , -0.1645718 , -0.07412154],  # Query 0
             [-0.14723456, -0.14723456, -0.14723456, -0.14723456, -0.08399412],  # Query 1
             [-0.16388056, -0.16388056, -0.16388056, -0.16388056, -0.05493092],  # Query 2
             [-0.1868437 , -0.1868437 , -0.1868437 , -0.1868437 , -0.07625321],  # Query 3
             [-0.20076165, -0.20076165, -0.20076165, -0.20076165, -0.0671524 ]]  # Query 4
            row_ind = [0, 1, 2, 3, 4]  # Which queries to use
            col_ind = [0, 4, 2, 3, 1]  # Which targets they match to
            This means:
                
                Query 0 matches with target 0 (class 1)
                Query 1 matches with target 4 (class 9)
                Query 2 matches with target 2 (class 1)
                Query 3 matches with target 3 (class 1)
                Query 4 matches with target 1 (class 1)
            Another Example which is easier:
            Ground truth labels tgt_ids[b] = [8, 0, 9, 1, 8] (5 targets)
            # Cost Matrix:
            [[-0.0751, -0.1023, -0.0876, -0.2142, -0.0751],  # Query 0
             [-0.0911, -0.0929, -0.0980, -0.2144, -0.0911],  # Query 1
             [-0.0606, -0.0813, -0.1075, -0.2602, -0.0606],  # Query 2
             [-0.0867, -0.1077, -0.1456, -0.1916, -0.0867],  # Query 3
             [-0.0803, -0.1265, -0.1192, -0.1970, -0.0803]]  # Query 4
            
                IMPORTANT:
                This cost matrix was created by selecting probabilities from out_prob based on tgt_ids:

                Column 0: probabilities for class 8 (first target)
                Column 1: probabilities for class 0 (second target)
                Column 2: probabilities for class 9 (third target)
                Column 3: probabilities for class 1 (fourth target)
                Column 4: probabilities for class 8 (fifth target)
                
                row_ind = [0, 1, 2, 3, 4]    # Predictions to use
                col_ind = [4, 0, 3, 2, 1]    # Which targets they match to
            """

        return indices

    def _get_layer_loss(self, pred, target, indices):
        """Helper to compute loss for a single layer's predictions"""
        losses = []
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            pred_i = pred[batch_idx][pred_idx]
            tgt_i = target[batch_idx][tgt_idx]
            loss = self.ce_loss(pred_i, tgt_i.argmax(-1))
            losses.append(loss.mean()) ### REMOVE MEAN HERE! UNNECESSARY
        return torch.stack(losses).mean()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: If auxiliary losses enabled: Tensor of shape [num_layers + 1, B, num_queries, num_classes]
                    Otherwise: Tensor of shape [B, num_queries, num_classes]
            targets: Tensor of shape [B, num_queries, num_classes]
        """
        # Check if we have auxiliary outputs
        if outputs.dim() == 4:  # Has auxiliary outputs [num_layers + 1, B, num_queries, num_classes]
            # Split predictions from different decoder layers
            aux_outputs = outputs[:-1]  # Predictions from intermediate layers
            outputs_final = outputs[-1]  # Predictions from final layer

            # Calculate matching using only the final layer predictions
            indices = self.Hungarian_matching(outputs_final, targets)

            # Calculate loss for final predictions using final layer matching
            final_loss = self._get_layer_loss(outputs_final, targets, indices)

            # Calculate auxiliary losses using the SAME indices from final layer
            aux_losses = []
            for aux_output in aux_outputs:
                # Use the same matching indices for all auxiliary layers
                layer_loss = self._get_layer_loss(aux_output, targets, indices)
                aux_losses.append(layer_loss)

            # Combine losses
            aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0)
            total_loss = final_loss + self.aux_loss_weight * aux_loss

            return total_loss

        else:  # No auxiliary outputs, just compute regular loss
            indices = self.Hungarian_matching(outputs, targets)
            return self._get_layer_loss(outputs, targets, indices)


class QuerySeparationLoss(nn.Module):
    """
    Supervised contrastive loss on the Hungarian-matched query embeddings.

    HungarianMatchingLoss decides *which* query predicts *which* activity but says nothing about the
    geometry of the query embeddings: the classification head can read one query for every activity
    and leave the representation entangled. This term pulls the matched-slot embeddings of
    same-activity samples together and pushes different activities apart, so the representation itself
    separates activities rather than leaving it to class_embed. 

    The anchors are the same matched pairs the supervised loss uses, so no second matching convention
    is introduced, and no-object targets are skipped (they are not an activity).

    Args:
        temperature: softmax temperature of the contrastive similarities (lower = sharper)
        no_object_index: column of the no-object class; None means the last class
    """

    def __init__(self, temperature=0.1, no_object_index=None):
        super().__init__()
        self.temperature = temperature
        self.no_object_index = no_object_index

    def forward(self, embeddings, targets, indices):
        """
        Args:
            embeddings: (B, num_queries, d) last-layer query embeddings
            targets: (B, num_queries, num_classes) one-hot target sets, no-object class last
            indices: the HungarianMatchingLoss indices: list of (row_ind, col_ind) per batch item,
                where row_ind indexes queries and col_ind the target slots they are matched to
        Returns:
            scalar loss; a differentiable zero when fewer than two anchors carry an activity
        """
        var_no_object = (self.no_object_index if self.no_object_index is not None
                         else targets.shape[-1] - 1)
        var_anchors, var_labels = [], []
        for var_b, (var_rows, var_cols) in enumerate(indices):
            for var_row, var_col in zip(var_rows.tolist(), var_cols.tolist()):
                var_class = int(targets[var_b, var_col].argmax())
                if var_class == var_no_object:
                    continue
                var_anchors.append(embeddings[var_b, var_row])
                var_labels.append(var_class)
        if len(var_anchors) < 2:
            return embeddings.sum() * 0.0
        var_z = F.normalize(torch.stack(var_anchors), dim=-1)
        var_labels = torch.as_tensor(var_labels, device=var_z.device)
        var_sim = (var_z @ var_z.t()) / self.temperature
        ## positive pairs are same-activity, different-sample; self-similarity is masked out
        var_self = torch.eye(var_z.shape[0], dtype=torch.bool, device=var_z.device)
        var_pos = (var_labels.unsqueeze(0) == var_labels.unsqueeze(1)) & ~var_self
        var_log_prob = var_sim - torch.logsumexp(var_sim, dim=1, keepdim=True)
        ## a class with a single anchor in the batch has no positive: it contributes 0, not a NaN
        return -((var_pos * var_log_prob).sum(1) / var_pos.sum(1).clamp_min(1)).mean()


class SetDistillationLoss(nn.Module):
    """
    Permutation-invariant distillation between a frozen teacher and a trainable student.

    Both networks are set predictors: their queries carry no fixed ordering, so a plain
    element-wise KL between teacher and student logits is meaningless. Instead the student's
    queries are matched to the teacher's queries with a Hungarian assignment computed on the
    final-layer class probabilities (the same cost used by HungarianMatchingLoss), and the
    matched pairs supervise every decoder layer with a temperature-scaled soft cross-entropy.

    Only the teacher's *real* queries take part in the assignment. A teacher whose head has
    collapsed onto a single query (what a single-occupant training set produces) emits one query
    that names an activity and num_queries - 1 that name no-object; matching the student's queries
    to those no-object queries with a full num_queries x num_queries assignment would supervise
    every other student query to no-object and re-impose the teacher's collapse, and the teacher's
    no-object mass would also suppress the classes it never saw. Restricting the assignment to the
    columns whose teacher argmax is not no-object leaves the student's unmatched queries free to be
    decided by the supervised Hungarian loss, which is what lets them spread over a frame's several
    occupants.

    Args:
        temperature: soft-target temperature
        aux_loss_weight: weight of the auxiliary decoder-layer terms
        cost_class_weight: weight of the classification cost in the assignment
        no_object_index: column of the no-object class; None means the last class (the repo's
            convention for build_activity_set_labels / reduce_dataset)
    """

    def __init__(self, temperature=1.0, aux_loss_weight=1.0, cost_class_weight=1.0,
                 no_object_index=None):
        super().__init__()
        self.temperature = temperature
        self.aux_loss_weight = aux_loss_weight
        self.cost_class = cost_class_weight
        self.no_object_index = no_object_index

    @torch.no_grad()
    def Hungarian_matching(self, student_final, teacher_final):
        """
        Match student queries to the teacher's real (non-no-object) queries.

        Args:
            student_final: [B, num_queries, num_classes] student final-layer logits
            teacher_final: [B, num_queries, num_classes] teacher final-layer logits
        Returns:
            List of (row_ind, col_ind) per batch item, where row_ind indexes student queries and
            col_ind the teacher queries they are matched to. A batch item the teacher predicts no
            object for yields an empty pair, i.e. no distillation for that sample.
        """
        bs, num_queries = student_final.shape[:2]
        no_object = (self.no_object_index if self.no_object_index is not None
                     else student_final.shape[-1] - 1)
        student_prob = student_final.softmax(-1)
        teacher_ids = teacher_final.argmax(-1)  # [B, num_queries] teacher pseudo-labels

        indices = []
        for b in range(bs):
            real = torch.nonzero(teacher_ids[b] != no_object, as_tuple=False).flatten()
            if len(real) == 0:
                empty = torch.empty(0, dtype=torch.int64, device=student_final.device)
                indices.append((empty, empty))
                continue
            cost_matrix = -student_prob[b][:, teacher_ids[b][real]]  # [student_q, num_real]
            cost_matrix = self.cost_class * cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=student_final.device)
            ## col_ind indexes the filtered columns; map it back to teacher query indices
            col_ind = torch.as_tensor(real.cpu().numpy()[col_ind], dtype=torch.int64,
                                      device=student_final.device)
            indices.append((row_ind, col_ind))

        return indices

    def _get_layer_loss(self, student_layer, teacher_layer, indices):
        """Temperature-scaled soft cross-entropy over the matched student/teacher queries."""
        temperature = self.temperature
        teacher_layer = teacher_layer.detach()
        losses = []
        for batch_idx, (student_idx, teacher_idx) in enumerate(indices):
            if len(student_idx) == 0:
                continue
            student_log_prob = F.log_softmax(student_layer[batch_idx][student_idx] / temperature, dim=-1)
            teacher_prob = F.softmax(teacher_layer[batch_idx][teacher_idx] / temperature, dim=-1)
            losses.append(-(teacher_prob * student_log_prob).sum(-1).mean())
        if not losses:
            ## no sample in the batch had a real teacher query: contribute a differentiable zero so
            ## the distillation term stays part of the graph (backward on a detached zero would fail
            ## when the supervised term is the only other contribution)
            return student_layer.sum() * 0.0
        return torch.stack(losses).mean() * (temperature ** 2)

    def forward(self, student_outputs, teacher_outputs):
        if student_outputs.dim() == 4:  # [num_layers, B, num_queries, num_classes]
            student_aux, student_final = student_outputs[:-1], student_outputs[-1]
            teacher_aux, teacher_final = teacher_outputs[:-1], teacher_outputs[-1]
        else:  # [B, num_queries, num_classes]
            student_aux, student_final = None, student_outputs
            teacher_aux, teacher_final = None, teacher_outputs

        indices = self.Hungarian_matching(student_final, teacher_final)

        final_loss = self._get_layer_loss(student_final, teacher_final, indices)

        if student_aux is None or len(student_aux) == 0:
            return final_loss

        aux_losses = [
            self._get_layer_loss(student_layer, teacher_layer, indices)
            for student_layer, teacher_layer in zip(student_aux, teacher_aux)
        ]
        aux_loss = torch.stack(aux_losses).mean()

        return final_loss + self.aux_loss_weight * aux_loss


class JointDistillationLoss(nn.Module):
    """
    Distillation between a frozen teacher and a trainable student for MultiSenseX.

    Unlike the AMAR set predictor, MultiSenseX has a fixed head layout: one activity head per
    location and one multi-label location head. There is no query permutation to resolve, so the
    soft targets are matched element-wise:
      - activity: temperature-scaled soft cross-entropy over the 9 classes of each location head
      - location: temperature-scaled Bernoulli cross-entropy on the 5 sigmoid outputs

    Args:
        student_outputs / teacher_outputs: (act_logits, loc_pred) where act_logits is
            (batch, 5, 9) and loc_pred is (batch, 5) sigmoid probabilities.
    """

    def __init__(self, temperature=1.0, activity_weight=1.0, location_weight=1.0, eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.activity_weight = activity_weight
        self.location_weight = location_weight
        self.eps = eps

    def forward(self, student_outputs, teacher_outputs):
        student_act, student_loc = student_outputs[0], student_outputs[1]
        teacher_act = teacher_outputs[0].detach()
        teacher_loc = teacher_outputs[1].detach()

        temperature = self.temperature

        # Activity: soft cross-entropy over the class dimension of every location head
        activity_kd = -(
            F.softmax(teacher_act / temperature, dim=-1)
            * F.log_softmax(student_act / temperature, dim=-1)
        ).sum(dim=-1).mean() * (temperature ** 2)

        # Location: Bernoulli soft cross-entropy, temperature applied in logit space
        student_logit = torch.logit(student_loc.clamp(self.eps, 1.0 - self.eps))
        teacher_prob = torch.sigmoid(
            torch.logit(teacher_loc.clamp(self.eps, 1.0 - self.eps)) / temperature
        )
        location_kd = F.binary_cross_entropy_with_logits(
            student_logit / temperature, teacher_prob
        ) * (temperature ** 2)

        return self.activity_weight * activity_kd + self.location_weight * location_kd


class OccupancyDistillationLoss(nn.Module):
    """
    Distillation between a frozen teacher and a trainable student density-map model.

    The density-map model has no query permutation: it emits one logit per grid cell of the spatial
    density map, so the soft targets are matched element-wise, exactly like the supervised map loss.
    The teacher's occupancy probability is the temperature-scaled soft target and the student is
    trained with a Bernoulli cross-entropy in logit space; the T^2 factor keeps the gradient
    magnitude comparable to the supervised term when the temperature changes.

    Accepts either the raw density logits or the density-map forward tuple
    (density, count, density_logits); the logits are the last element.

    Args:
        temperature: soft-target temperature applied in logit space
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _logits(outputs):
        if isinstance(outputs, (tuple, list)):
            return outputs[-1]
        return outputs

    def forward(self, student_outputs, teacher_outputs):
        student_logits = self._logits(student_outputs)
        teacher_logits = self._logits(teacher_outputs).detach()

        temperature = self.temperature
        teacher_prob = torch.sigmoid(teacher_logits / temperature)

        return F.binary_cross_entropy_with_logits(
            student_logits / temperature, teacher_prob
        ) * (temperature ** 2)
