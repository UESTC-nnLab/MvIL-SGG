import numpy as np
import torch # 或者移除torch相关，如果你的logits是纯numpy

def logits_to_scores(logits):
    """
    Converts logits to scores. For simplicity, we can use logits directly
    as scores since their ranking order is preserved by monotonic functions
    like softmax. Alternatively, apply softmax if probabilities are desired
    for specific scoring schemes (e.g., product of probabilities).
    Using logits directly means we'll sum them for combined scores.
    """
    if isinstance(logits, torch.Tensor):
        return logits.detach().cpu().numpy()
    return logits

def calculate_recall_at_k(gt_relationships, predicted_relationships_sorted, k_values, is_sgcls=False, gt_object_labels=None):
    """
    Calculates Recall@K for given K values.

    Args:
        gt_relationships (set): A set of ground truth relationship tuples.
            For PredCls: {(subj_idx, obj_idx, pred_label_idx), ...}
            For SGCls: {(subj_idx, obj_idx, subj_gt_label, obj_gt_label, pred_label_idx), ...}
        predicted_relationships_sorted (list): A list of predicted relationship tuples,
            sorted by confidence score in descending order.
            Each tuple contains the predicted relationship and its score:
            For PredCls: [((subj_idx, obj_idx, pred_label_idx), score), ...]
            For SGCls: [((subj_idx, obj_idx, subj_pred_label, obj_pred_label, pred_label_idx), score), ...]
        k_values (list): A list of integers for K, e.g., [20, 50, 100].
        is_sgcls (bool): Flag to indicate if the calculation is for SGCls.
        gt_object_labels (np.array, optional): Ground truth object labels, required if is_sgcls=True
                                                to construct the full GT triplet for SGCls.

    Returns:
        dict: A dictionary with K as key and Recall@K as value.
              e.g., {20: 0.5, 50: 0.6, 100: 0.7}
        int: Number of ground truth relationships.
    """
    results = {k: 0.0 for k in k_values}
    max_k = max(k_values)

    if not gt_relationships: # No ground truth relationships
        for k in k_values:
            results[k] = 1.0 # Conventionally, R@K is 1.0 if there are no GTs to recall
        return results, 0

    recalled_gt_triplets = set() # To avoid double counting the same GT triplet

    for i, (pred_triplet_elements, score) in enumerate(predicted_relationships_sorted):
        if i >= max_k and not all(len(recalled_gt_triplets) == len(gt_relationships) for k_val in k_values): # Optimization
             # if i >= max_k and len(recalled_gt_triplets) == len(gt_relationships): # if all GTs already found
             # break # No need to check further than max_K if all GTs found
            pass # Must iterate up to max_k for all k values unless all GTs are found earlier for all K.

        # The pred_triplet_elements are what we check against the gt_relationships set
        if pred_triplet_elements in gt_relationships:
            recalled_gt_triplets.add(pred_triplet_elements)

        for k in k_values:
            if i < k: # Check if current prediction is within top-k
                      # This logic is slightly off. We need to count distinct GTs found *within* the top k.
                pass # Corrected logic below.

    # Corrected R@K calculation:
    # After iterating through all predictions (or up to max_k),
    # or by iterating through top-k for each k.
    # The loop above collects *all* recalled GTs from the sorted list.
    # Let's refine:
    
    num_gt = len(gt_relationships)
    recalled_counts_at_k = {k: 0 for k in k_values}
    # This set stores GTs that have been recalled by *any* top-k prediction for that k.
    # A GT triplet should only be counted once per K.
    
    already_recalled_for_k = {k: set() for k in k_values}

    for rank, (pred_triplet_elements, score) in enumerate(predicted_relationships_sorted):
        # pred_triplet_elements is the key to check in gt_relationships
        is_correct_prediction = pred_triplet_elements in gt_relationships
        
        if is_correct_prediction:
            for k_val in k_values:
                if rank < k_val: # If this prediction is within the top k_val
                    # Add the *ground truth triplet* that was matched.
                    # Since pred_triplet_elements *is* the GT triplet form here, we add it.
                    already_recalled_for_k[k_val].add(pred_triplet_elements)
    
    for k_val in k_values:
        results[k_val] = len(already_recalled_for_k[k_val]) / num_gt if num_gt > 0 else 1.0

    return results, num_gt


def evaluate_predcls_batch(relationship_logits, gt_relationships_indices, k_values=[20, 50, 100]):
    """
    Evaluates Predicate Classification (PredCls) for a single batch/clip.

    Args:
        relationship_logits (np.array): Shape (N, N, num_predicate_classes).
        gt_relationships_indices (list): List of tuples (subj_idx, obj_idx, pred_label_idx).
        k_values (list): List of K values for R@K.

    Returns:
        dict: R@K results for the batch.
        int: Number of GT relationships in the batch.
    """
    relationship_logits = logits_to_scores(relationship_logits)
    N = relationship_logits.shape[0]
    num_predicate_classes = relationship_logits.shape[2]

    # Prepare GT set for quick lookup
    # For PredCls, GT is (subj_idx, obj_idx, pred_label_idx)
    gt_rel_set = set()
    for subj_idx, obj_idx, pred_label_idx in gt_relationships_indices:
        if subj_idx != obj_idx : # Ensure not self-loop unless allowed
             gt_rel_set.add((subj_idx, obj_idx, pred_label_idx))

    # Generate all possible predicted relationships and their scores
    predicted_relationships_with_scores = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for pred_label_idx in range(num_predicate_classes):
                score = relationship_logits[i, j, pred_label_idx]
                # For PredCls, the triplet is (subj_idx, obj_idx, pred_idx)
                predicted_relationships_with_scores.append(
                    ((i, j, pred_label_idx), score)
                )

    # Sort predictions by score
    predicted_relationships_with_scores.sort(key=lambda x: x[1], reverse=True)

    return calculate_recall_at_k(gt_rel_set, predicted_relationships_with_scores, k_values)


def evaluate_sgcls_batch(object_logits, relationship_logits, gt_object_labels, gt_relationships_indices, k_values=[20, 50, 100]):
    """
    Evaluates Scene Graph Classification (SGCls) for a single batch/clip.

    Args:
        object_logits (np.array): Shape (N, num_object_classes).
        relationship_logits (np.array): Shape (N, N, num_predicate_classes).
        gt_object_labels (np.array): Shape (N,), true object class indices.
        gt_relationships_indices (list): List of tuples (subj_idx, obj_idx, pred_label_idx).
        k_values (list): List of K values for R@K.

    Returns:
        dict: R@K results for the batch.
        int: Number of GT relationships in the batch.
    """
    object_logits = logits_to_scores(object_logits)
    relationship_logits = logits_to_scores(relationship_logits)
    gt_object_labels = logits_to_scores(gt_object_labels) # ensure numpy

    N = object_logits.shape[0]
    num_predicate_classes = relationship_logits.shape[2]

    # Predict object labels (highest logit)
    pred_object_labels = np.argmax(object_logits, axis=1)

    # Prepare GT set for SGCls
    # For SGCls, GT triplet is (subj_idx, obj_idx, subj_gt_label, obj_gt_label, pred_gt_label)
    gt_triplet_set = set()
    for subj_idx, obj_idx, pred_label_idx in gt_relationships_indices:
        if subj_idx != obj_idx:
            subj_gt_label = gt_object_labels[subj_idx]
            obj_gt_label = gt_object_labels[obj_idx]
            gt_triplet_set.add((subj_idx, obj_idx, subj_gt_label, obj_gt_label, pred_label_idx))

    # Generate all possible predicted triplets and their combined scores
    predicted_triplets_with_scores = []
    for i in range(N): # Subject
        for j in range(N): # Object
            if i == j:
                continue

            # Predicted subject and object labels and their logits (scores)
            subj_pred_label = pred_object_labels[i]
            obj_pred_label = pred_object_labels[j]
            
            score_subj_logit = object_logits[i, subj_pred_label]
            score_obj_logit = object_logits[j, obj_pred_label]

            for pred_label_idx in range(num_predicate_classes):
                score_rel_logit = relationship_logits[i, j, pred_label_idx]
                
                # Combined score: sum of logits
                # (Using sum of logits is common. If using probabilities, it would be a product)
                combined_score = score_subj_logit + score_obj_logit + score_rel_logit
                
                # For SGCls, the triplet to match is (subj_idx, obj_idx, subj_pred_label, obj_pred_label, pred_idx)
                # The indices i and j are the original object indices from the input.
                predicted_triplets_with_scores.append(
                    ((i, j, subj_pred_label, obj_pred_label, pred_label_idx), combined_score)
                )
    
    # Sort predictions by combined_score
    predicted_triplets_with_scores.sort(key=lambda x: x[1], reverse=True)

    return calculate_recall_at_k(gt_triplet_set, predicted_triplets_with_scores, k_values, is_sgcls=True, gt_object_labels=gt_object_labels)


# --- Mock Data (5 objects) ---
# GT Object Labels: [7 6 0 7 9]
# GT Relationships: [(0, 4, 1), (4, 1, 0), (0, 4, 6), (4, 3, 0), (4, 0, 2)]
# Object Logits shape: (5, 10)
# Relationship Logits shape: (5, 5, 7)
# -------------------------------------

# --- Example Usage ---
if __name__ == '__main__':
    # Mock data for one batch/clip
    N_objects = 5
    N_object_classes = 10
    N_predicate_classes = 7
    K_VALUES = [20, 50, 100] # R@K values to calculate, ensure K is not too large for small N

    # Ground Truth
    gt_obj_labels = np.random.randint(0, N_object_classes, size=N_objects)
    # gt_relationships: (subj_idx, obj_idx, predicate_label_idx)
    # Ensure subj_idx != obj_idx for simplicity in mock data
    gt_rels = []
    num_gt_rels_to_generate = N_objects * (N_objects -1) // 4 # Generate some random relations
    for _ in range(num_gt_rels_to_generate):
        s = np.random.randint(0,N_objects)
        o = np.random.randint(0,N_objects)
        if s == o : continue
        p = np.random.randint(0,N_predicate_classes)
        gt_rels.append((s,o,p))
    gt_rels = list(set(gt_rels)) # Ensure unique gt relations

    # Model Predictions (logits)
    # Replace with your actual model outputs
    mock_obj_logits = np.random.rand(N_objects, N_object_classes) * 5 # Simulate logits
    mock_rel_logits = np.random.rand(N_objects, N_objects, N_predicate_classes) * 5 # Simulate logits

    print(f"--- Mock Data ({N_objects} objects) ---")
    print(f"GT Object Labels: {gt_obj_labels}")
    print(f"GT Relationships: {gt_rels}")
    print(f"Object Logits shape: {mock_obj_logits.shape}")
    print(f"Relationship Logits shape: {mock_rel_logits.shape}")
    print("-------------------------------------\n")

    # --- PredCls Evaluation ---
    print("--- PredCls Evaluation (Batch) ---")
    # For PredCls, the model is *given* gt_obj_labels, but they are not directly part of the predicted triplet key
    # The gt_rel_set for predcls is (subj_idx, obj_idx, pred_label_idx)
    predcls_results, num_gt_predcls = evaluate_predcls_batch(mock_rel_logits, gt_rels, K_VALUES)
    print(f"Number of GT relationships for PredCls: {num_gt_predcls}")
    for k, recall in predcls_results.items():
        print(f"PredCls R@{k}: {recall:.4f}")
    print("-------------------------------------\n")

    # --- SGCls Evaluation ---
    print("--- SGCls Evaluation (Batch) ---")
    sgcls_results, num_gt_sgcls = evaluate_sgcls_batch(mock_obj_logits, mock_rel_logits, gt_obj_labels, gt_rels, K_VALUES)
    print(f"Number of GT relationships for SGCls: {num_gt_sgcls}")
    for k, recall in sgcls_results.items():
        print(f"SGCls R@{k}: {recall:.4f}")
    print("-------------------------------------\n")

    # --- To run over an entire dataset ---
    # You would typically loop through your data loader:
    # total_recalled_at_k_predcls = {k: 0 for k in K_VALUES}
    # total_gt_rels_predcls = 0
    # total_recalled_at_k_sgcls = {k: 0 for k in K_VALUES}
    # total_gt_rels_sgcls = 0
    #
    # for batch_data in test_loader:
    #     # Extract obj_logits, rel_logits, gt_obj_labels, gt_rels from batch_data
    #     obj_logits, rel_logits, gt_obj_labels_b, gt_relationships_indices_b = ...
    #
    #     # PredCls
    #     predcls_res_b, num_gt_b_pc = evaluate_predcls_batch(rel_logits, gt_relationships_indices_b, K_VALUES)
    #     if num_gt_b_pc > 0: # Only accumulate if there were GTs to recall
    #         for k in K_VALUES:
    #             total_recalled_at_k_predcls[k] += predcls_res_b[k] * num_gt_b_pc # recall * num_gt = num_recalled
    #         total_gt_rels_predcls += num_gt_b_pc
    #
    #     # SGCls
    #     sgcls_res_b, num_gt_b_sgc = evaluate_sgcls_batch(obj_logits, rel_logits, gt_obj_labels_b, gt_relationships_indices_b, K_VALUES)
    #     if num_gt_b_sgc > 0:
    #         for k in K_VALUES:
    #             total_recalled_at_k_sgcls[k] += sgcls_res_b[k] * num_gt_b_sgc
    #         total_gt_rels_sgcls += num_gt_b_sgc
    #
    # print("--- Overall Dataset Results ---")
    # print("PredCls:")
    # for k in K_VALUES:
    #     overall_r_at_k = total_recalled_at_k_predcls[k] / total_gt_rels_predcls if total_gt_rels_predcls > 0 else 0
    #     print(f"  Overall PredCls R@{k}: {overall_r_at_k:.4f}")
    #
    # print("SGCls:")
    # for k in K_VALUES:
    #     overall_r_at_k = total_recalled_at_k_sgcls[k] / total_gt_rels_sgcls if total_gt_rels_sgcls > 0 else 0
    #     print(f"  Overall SGCls R@{k}: {overall_r_at_k:.4f}")
    # print("-------------------------------")