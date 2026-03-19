import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, auc, precision_recall_curve, roc_curve
import random
import os

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

patch_size = 4
len_model = int(160 // patch_size)

all_data_np = np.load(r'all_DTWsim_data.npy')
all_labels_np = np.load(r'all_DTWsim_label.npy')
own_test_data = np.load(r'test_data.npy')
own_test_label = np.load(r'test_label.npy')


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, normal_features, anomaly_features):
        similarity = self.cos_sim(normal_features, anomaly_features) / self.temperature
        loss = torch.mean(-torch.log(1 - torch.sigmoid(similarity) + 1e-8))
        return loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ContrastiveTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(ContrastiveTimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1 * patch_size)
        self.sigmoid = nn.Sigmoid()
        self.fragment_expansion = nn.Linear(13, 160)
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )

    def forward(self, x, get_features=False):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        if seq_len == 160:
            x_patched = x.view(batch_size, len_model, patch_size, -1).view(batch_size, len_model, -1)
        elif seq_len == 13:
            x = self.fragment_expansion(x.transpose(1, 2)).transpose(1, 2)
            x_patched = x.view(batch_size, len_model, patch_size, -1).view(batch_size, len_model, -1)
        else:
            raise ValueError(f"Unexpected sequence length: {seq_len}")
        x = self.embedding(x_patched)
        x = self.pos_encoder(x)
        features = self.transformer_encoder(x)
        if get_features:
            global_features = torch.mean(features, dim=1)
            global_features = self.feature_extractor(global_features)
            return global_features
        x = self.output_layer(features)
        x = x.view(batch_size, len_model, patch_size, -1).view(batch_size, 160, -1)
        x = self.sigmoid(x)
        return x.squeeze(-1)


def extract_fragments(data, labels, fragment_len=13):
    n_samples = data.shape[0]
    seq_len = data.shape[1]
    normal_fragments = np.zeros((n_samples, fragment_len, data.shape[2]))
    anomaly_fragments = np.zeros((n_samples, fragment_len, data.shape[2]))
    for i in range(n_samples):
        anomaly_indices = np.where(labels[i] == 1)[0]
        if len(anomaly_indices) == 0:
            anomaly_start = np.random.randint(0, seq_len - fragment_len)
            anomaly_end = anomaly_start + fragment_len
            normal_start = (anomaly_end + fragment_len) % seq_len
            normal_end = normal_start + fragment_len
            if normal_end > seq_len:
                normal_start = max(0, seq_len - fragment_len)
                normal_end = seq_len
        else:
            anomaly_start = anomaly_indices[0]
            anomaly_end = anomaly_indices[-1] + 1
            anomaly_center = (anomaly_start + anomaly_end) // 2
            half_len = fragment_len // 2
            anomaly_start = max(0, anomaly_center - half_len)
            anomaly_end = min(seq_len, anomaly_start + fragment_len)
            if anomaly_end - anomaly_start < fragment_len:
                anomaly_start = max(0, anomaly_end - fragment_len)
            if anomaly_start >= fragment_len:
                normal_start = anomaly_start - fragment_len
                normal_end = anomaly_start
            else:
                normal_start = min(anomaly_end, seq_len - fragment_len)
                normal_end = normal_start + fragment_len
        normal_end = min(normal_end, seq_len)
        anomaly_end = min(anomaly_end, seq_len)
        if normal_end - normal_start == fragment_len:
            normal_fragments[i] = data[i, normal_start:normal_end]
        else:
            actual_len = normal_end - normal_start
            normal_fragments[i, :actual_len] = data[i, normal_start:normal_end]
        if anomaly_end - anomaly_start == fragment_len:
            anomaly_fragments[i] = data[i, anomaly_start:anomaly_end]
        else:
            actual_len = anomaly_end - anomaly_start
            anomaly_fragments[i, :actual_len] = data[i, anomaly_start:anomaly_end]
    return normal_fragments, anomaly_fragments


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


def evaluate(model, dataloader, threshold=0.1):
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


def train_model_with_contrastive(model, train_loader, valid_loader, optimizer, criterion, contrastive_criterion,
                                 normal_fragments, anomaly_fragments, num_epochs=30, contrastive_weight=0.5):
    best_f1 = 0
    best_model_state = None
    normal_tensor = torch.FloatTensor(normal_fragments)
    anomaly_tensor = torch.FloatTensor(anomaly_fragments)
    fragment_dataset = TensorDataset(normal_tensor, anomaly_tensor)
    fragment_loader = DataLoader(fragment_dataset, batch_size=16, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        contrastive_loss_total = 0
        fragment_iter = iter(fragment_loader)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            main_loss = criterion(outputs, labels)
            try:
                normal_batch, anomaly_batch = next(fragment_iter)
            except StopIteration:
                fragment_iter = iter(fragment_loader)
                normal_batch, anomaly_batch = next(fragment_iter)
            normal_batch, anomaly_batch = normal_batch.to(device), anomaly_batch.to(device)
            normal_features = model(normal_batch, get_features=True)
            anomaly_features = model(anomaly_batch, get_features=True)
            contrast_loss = contrastive_criterion(normal_features, anomaly_features)
            loss = main_loss + contrastive_weight * contrast_loss
            loss.backward()
            optimizer.step()
            train_loss += main_loss.item()
            contrastive_loss_total += contrast_loss.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_contrastive_loss = contrastive_loss_total / len(train_loader)
        metrics, _, _ = evaluate(model, valid_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
                  f"Valid Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                  f"F1: {metrics['f1_score']:.4f}, False Alarm: {metrics['false_alarm']:.4f}")
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_state = model.state_dict().copy()
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def main_contrastive(number_epoch=100, contrastive_weight=0.5):
    import math
    X_train = all_data_np[:1501, :, :]
    y_train = all_labels_np[:1501, :]
    X_valid = all_data_np[1501:2001, :, :]
    y_valid = all_labels_np[1501:2001, :]
    X_test = own_test_data
    y_test = own_test_label
    print("Extracting fragments for contrastive learning...")
    normal_fragments, anomaly_fragments = extract_fragments(X_train, y_train, fragment_len=13)
    print(f"Fragment shapes: normal {normal_fragments.shape}, anomaly {anomaly_fragments.shape}")
    train_inputs = torch.FloatTensor(X_train)
    train_labels = torch.FloatTensor(y_train)
    valid_inputs = torch.FloatTensor(X_valid)
    valid_labels = torch.FloatTensor(y_valid)
    test_inputs = torch.FloatTensor(X_test)
    test_labels = torch.FloatTensor(y_test)
    batch_size = 16
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_inputs, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = ContrastiveTimeSeriesTransformer(
        input_dim=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.2
    ).to(device)
    criterion = nn.BCELoss()
    contrastive_criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Starting training, contrastive weight: {contrastive_weight}")
    model = train_model_with_contrastive(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        normal_fragments=normal_fragments,
        anomaly_fragments=anomaly_fragments,
        num_epochs=number_epoch,
        contrastive_weight=contrastive_weight
    )
    test_metrics, test_predictions, test_scores = evaluate(model, test_loader)
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
    import os
    save_path = 'own_results\contrastive_transformer'
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + '/own_predictions.npy', test_predictions)
    np.save(save_path + '/own_scores.npy', test_scores)
    print(f"Predictions saved to {save_path}")
    return model, test_metrics


if __name__ == "__main__":
    model, metrics = main_contrastive(number_epoch=200, contrastive_weight=0.5)
