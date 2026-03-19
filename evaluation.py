import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, auc, precision_recall_curve, roc_curve
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ranges(indices):
    if len(indices) == 0:
        return []
    ranges = []
    start = indices[0]
    end = indices[0]
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append((start, end + 1))
            start = indices[i]
            end = indices[i]
    ranges.append((start, end + 1))
    return ranges


def evaluate_stock(model, dataloader, threshold=0.1):
    model.eval()
    all_preds = []
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = (outputs >= threshold).float()
            scores = outputs
            all_preds.append(predictions.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    flat_preds = all_preds.reshape(-1)
    flat_labels = all_labels.reshape(-1)
    flat_scores = all_scores.reshape(-1)
    precision = precision_score(flat_labels, flat_preds, zero_division=0)
    recall = recall_score(flat_labels, flat_preds, zero_division=0)
    f1 = f1_score(flat_labels, flat_preds, zero_division=0)
    false_positives = np.sum((flat_preds == 1) & (flat_labels == 0))
    true_negatives = np.sum((flat_preds == 0) & (flat_labels == 0))
    false_alarm = false_positives / (false_positives + true_negatives + 1e-10)
    accuracy = np.mean(flat_preds == flat_labels)
    mcc = matthews_corrcoef(flat_labels, flat_preds)
    fpr, tpr, _ = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(flat_labels, flat_scores)
    pr_auc = auc(recall_curve, precision_curve)
    batch_size = all_preds.shape[0]
    sample_ious = []
    detection_delays = []
    tp_scores = []
    for i in range(batch_size):
        pred_anomaly = np.where(all_preds[i] == 1)[0]
        true_anomaly = np.where(all_labels[i] == 1)[0]
        if len(pred_anomaly) > 0 and len(true_anomaly) > 0:
            pred_ranges = get_ranges(pred_anomaly)
            true_ranges = get_ranges(true_anomaly)
            max_iou = 0
            for pr in pred_ranges:
                for tr in true_ranges:
                    intersection = max(0, min(pr[1], tr[1]) - max(pr[0], tr[0]))
                    if intersection > 0:
                        union = (pr[1] - pr[0]) + (tr[1] - tr[0]) - intersection
                        iou = intersection / union
                        max_iou = max(max_iou, iou)
            sample_ious.append(max_iou)
            earliest_true = true_anomaly[0]
            earliest_pred = min(pred_anomaly) if len(pred_anomaly) > 0 else float('inf')
            if earliest_pred < float('inf'):
                delay = max(0, earliest_pred - earliest_true)
                detection_delays.append(delay)
                alpha = 0.5
                score = np.exp(-alpha * delay / len(all_labels[i]))
                tp_scores.append(score)
        elif len(pred_anomaly) == 0 and len(true_anomaly) == 0:
            sample_ious.append(1.0)
        else:
            sample_ious.append(0.0)
            if len(true_anomaly) > 0 and len(pred_anomaly) == 0:
                detection_delays.append(len(all_labels[i]))
                tp_scores.append(0.0)
    mean_iou = np.mean(sample_ious) if sample_ious else 0.0
    mean_detection_delay = np.mean(detection_delays) if detection_delays else float('inf')
    early_detection_score = np.mean(tp_scores) if tp_scores else 0.0
    total_anomalies = np.sum(flat_labels)
    correct_anomalies = np.sum((flat_preds == 1) & (flat_labels == 1))
    coverage = correct_anomalies / (total_anomalies + 1e-10)
    miss_rate = 1.0 - coverage
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_alarm': false_alarm,
        'accuracy': accuracy,
        'iou': mean_iou,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mean_detection_delay': mean_detection_delay,
        'early_detection_score': early_detection_score,
        'coverage': coverage,
        'miss_rate': miss_rate
    }, all_preds, all_scores


def save_evaluate_stock(model, X_test, y_test, save_path=None):
    test_inputs = torch.FloatTensor(X_test)
    test_labels = torch.FloatTensor(y_test)
    batch_size = 16
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_metrics, test_predictions, test_scores = evaluate_stock(model, test_loader)
    print("\nTest Results:")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"False Alarm Rate: {test_metrics['false_alarm']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")
    print(f"Matthews(MCC): {test_metrics['mcc']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Mean Detection Delay: {test_metrics['mean_detection_delay']:.4f}")
    print(f"Early Detection Score: {test_metrics['early_detection_score']:.4f}")
    print(f"Coverage: {test_metrics['coverage']:.4f}")
    print(f"Miss Rate: {test_metrics['miss_rate']:.4f}")
    if save_path != None:
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path + '/predictions.npy', test_predictions)
        np.save(save_path + '/scores.npy', test_scores)
        print(f"Predictions saved to {save_path}")


def reciprocal_rank(y_true, y_score):
    ranked_indices = np.argsort(y_score)[::-1]
    true_anomaly_indices = np.where(y_true == 1)[0]
    if len(true_anomaly_indices) == 0:
        return 0
    true_idx = true_anomaly_indices[0]
    rank = np.where(ranked_indices == true_idx)[0][0] + 1
    return 1.0 / rank


def mean_reciprocal_rank(y_true_list, y_score_list):
    rr_scores = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        rr = reciprocal_rank(y_true, y_score)
        rr_scores.append(rr)
    return np.mean(rr_scores) if rr_scores else 0


def hit_rate_at_k(y_true_list, y_score_list, k):
    hits = 0
    for y_true, y_score in zip(y_true_list, y_score_list):
        top_k_indices = np.argsort(y_score)[-k:]
        true_anomaly_indices = np.where(y_true == 1)[0]
        if len(true_anomaly_indices) == 0:
            continue
        true_idx = true_anomaly_indices[0]
        if true_idx in top_k_indices:
            hits += 1
    return hits / len(y_true_list) if len(y_true_list) > 0 else 0


def ndcg_at_k(y_true, y_score, k):
    ranked_indices = np.argsort(y_score)[::-1][:k]
    dcg = 0
    for i, idx in enumerate(ranked_indices):
        if y_true[idx] == 1:
            dcg += 1.0 / np.log2(i + 2)
    num_relevant = min(k, np.sum(y_true == 1))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    if idcg == 0:
        return 0
    return dcg / idcg


def mean_ndcg_at_k(y_true_list, y_score_list, k):
    ndcg_scores = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        ndcg = ndcg_at_k(y_true, y_score, k)
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores) if ndcg_scores else 0


def mean_rank(y_true_list, y_score_list):
    ranks = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        ranked_indices = np.argsort(y_score)[::-1]
        true_anomaly_indices = np.where(y_true == 1)[0]
        if len(true_anomaly_indices) == 0:
            continue
        true_idx = true_anomaly_indices[0]
        rank = np.where(ranked_indices == true_idx)[0][0] + 1
        ranks.append(rank)
    return np.mean(ranks) if ranks else float('inf')


def evaluate_crypto(model, dataloader, threshold=0.1, k_values=[5, 10, 20]):
    model.eval()
    all_preds = []
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = (outputs >= threshold).float()
            scores = outputs
            all_preds.append(predictions.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    flat_preds = all_preds.reshape(-1)
    flat_labels = all_labels.reshape(-1)
    flat_scores = all_scores.reshape(-1)
    precision = precision_score(flat_labels, flat_preds, zero_division=0)
    recall = recall_score(flat_labels, flat_preds, zero_division=0)
    f1 = f1_score(flat_labels, flat_preds, zero_division=0)
    false_positives = np.sum((flat_preds == 1) & (flat_labels == 0))
    true_negatives = np.sum((flat_preds == 0) & (flat_labels == 0))
    false_alarm = false_positives / (false_positives + true_negatives + 1e-10)
    accuracy = np.mean(flat_preds == flat_labels)
    mcc = matthews_corrcoef(flat_labels, flat_preds)
    fpr, tpr, _ = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(flat_labels, flat_scores)
    pr_auc = auc(recall_curve, precision_curve)
    batch_size = all_preds.shape[0]
    sample_ious = []
    detection_delays = []
    tp_scores = []
    for i in range(batch_size):
        pred_anomaly = np.where(all_preds[i] == 1)[0]
        true_anomaly = np.where(all_labels[i] == 1)[0]
        if len(pred_anomaly) > 0 and len(true_anomaly) > 0:
            pred_ranges = get_ranges(pred_anomaly)
            true_ranges = get_ranges(true_anomaly)
            max_iou = 0
            for pr in pred_ranges:
                for tr in true_ranges:
                    intersection = max(0, min(pr[1], tr[1]) - max(pr[0], tr[0]))
                    if intersection > 0:
                        union = (pr[1] - pr[0]) + (tr[1] - tr[0]) - intersection
                        iou = intersection / union
                        max_iou = max(max_iou, iou)
            sample_ious.append(max_iou)
            earliest_true = true_anomaly[0]
            earliest_pred = min(pred_anomaly) if len(pred_anomaly) > 0 else float('inf')
            if earliest_pred < float('inf'):
                delay = max(0, earliest_pred - earliest_true)
                detection_delays.append(delay)
                alpha = 0.5
                score = np.exp(-alpha * delay / len(all_labels[i]))
                tp_scores.append(score)
        elif len(pred_anomaly) == 0 and len(true_anomaly) == 0:
            sample_ious.append(1.0)
        else:
            sample_ious.append(0.0)
            if len(true_anomaly) > 0 and len(pred_anomaly) == 0:
                detection_delays.append(len(all_labels[i]))
                tp_scores.append(0.0)
    mean_iou = np.mean(sample_ious) if sample_ious else 0.0
    mean_detection_delay = np.mean(detection_delays) if detection_delays else float('inf')
    early_detection_score = np.mean(tp_scores) if tp_scores else 0.0
    total_anomalies = np.sum(flat_labels)
    correct_anomalies = np.sum((flat_preds == 1) & (flat_labels == 1))
    coverage = correct_anomalies / (total_anomalies + 1e-10)
    miss_rate = 1.0 - coverage
    y_true_list = [all_labels[i] for i in range(batch_size)]
    y_score_list = [all_scores[i] for i in range(batch_size)]
    mrr = mean_reciprocal_rank(y_true_list, y_score_list)
    avg_rank = mean_rank(y_true_list, y_score_list)
    hit_rates = {}
    ndcg_scores = {}
    for k in k_values:
        hit_rates[f'hit_rate@{k}'] = hit_rate_at_k(y_true_list, y_score_list, k)
        ndcg_scores[f'ndcg@{k}'] = mean_ndcg_at_k(y_true_list, y_score_list, k)
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_alarm': false_alarm,
        'accuracy': accuracy,
        'iou': mean_iou,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mean_detection_delay': mean_detection_delay,
        'early_detection_score': early_detection_score,
        'coverage': coverage,
        'miss_rate': miss_rate,
        'mrr': mrr,
        'mean_rank': avg_rank,
    }
    metrics.update(hit_rates)
    metrics.update(ndcg_scores)
    return metrics, all_preds, all_scores


def save_evaluate_crypto(model, X_test, y_test, save_path=None, k_values=[5, 10, 20]):
    test_inputs = torch.FloatTensor(X_test)
    test_labels = torch.FloatTensor(y_test)
    batch_size = 16
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_metrics, test_predictions, test_scores = evaluate_crypto(
        model, test_loader, k_values=k_values
    )
    return test_metrics, test_predictions, test_scores
