import torch
import torch.nn as nn
import numpy as np
import time
import os
import tempfile

patch_size = 4
len_model  = int(160 // patch_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ContrastiveTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4,
                 num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1 * patch_size)
        self.sigmoid = nn.Sigmoid()
        self.fragment_expansion = nn.Linear(13, 160)
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4))

    def forward(self, x, get_features=False):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        if seq_len == 160:
            x_patched = x.view(batch_size, len_model, patch_size, -1)\
                         .view(batch_size, len_model, -1)
        elif seq_len == 13:
            x = self.fragment_expansion(x.transpose(1, 2)).transpose(1, 2)
            x_patched = x.view(batch_size, len_model, patch_size, -1)\
                         .view(batch_size, len_model, -1)
        else:
            raise ValueError(f"Unexpected sequence length: {seq_len}")
        x = self.embedding(x_patched)
        x = self.pos_encoder(x)
        features = self.transformer_encoder(x)
        if get_features:
            gf = torch.mean(features, dim=1)
            return self.feature_extractor(gf)
        x = self.output_layer(features)
        x = x.view(batch_size, len_model, patch_size, -1).view(batch_size, 160, -1)
        return self.sigmoid(x).squeeze(-1)


model = ContrastiveTimeSeriesTransformer(
    input_dim=1, d_model=64, nhead=4,
    num_encoder_layers=2, dim_feedforward=128, dropout=0.2)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*55}")
print(f"  Total parameters    : {total_params:,}  ({total_params/1e6:.4f} M)")
print(f"  Trainable parameters: {trainable_params:,}  ({trainable_params/1e6:.4f} M)")

with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
    tmp_path = tmp.name
torch.save(model.state_dict(), tmp_path)
disk_mb = os.path.getsize(tmp_path) / 1024**2
os.remove(tmp_path)
print(f"  Model file size     : {disk_mb:.2f} MB  (state_dict on disk)")

WARMUP = 50
REPEATS = 1000
dummy = torch.randn(1, 160, 1)

with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(dummy)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        _ = model(dummy)
    t1 = time.perf_counter()

cpu_ms_single = (t1 - t0) / REPEATS * 1000
print(f"\n  CPU inference (batch=1)  : {cpu_ms_single:.3f} ms / window")

dummy_batch = torch.randn(16, 160, 1)
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(dummy_batch)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        _ = model(dummy_batch)
    t1 = time.perf_counter()

cpu_ms_batch = (t1 - t0) / REPEATS * 1000
print(f"  CPU inference (batch=16) : {cpu_ms_batch:.3f} ms / batch  "
      f"({cpu_ms_batch/16:.3f} ms / window)")

if torch.cuda.is_available():
    device = torch.device('cuda')
    model_gpu = model.to(device)
    dummy_gpu = torch.randn(1, 160, 1).to(device)
    dummy_b_gpu = torch.randn(16, 160, 1).to(device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model_gpu(dummy_gpu)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start_event.record()
        for _ in range(REPEATS):
            _ = model_gpu(dummy_gpu)
        end_event.record()
    torch.cuda.synchronize()
    gpu_ms_single = start_event.elapsed_time(end_event) / REPEATS
    print(f"  GPU inference (batch=1)  : {gpu_ms_single:.3f} ms / window")

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model_gpu(dummy_b_gpu)
    torch.cuda.synchronize()
    with torch.no_grad():
        start_event.record()
        for _ in range(REPEATS):
            _ = model_gpu(dummy_b_gpu)
        end_event.record()
    torch.cuda.synchronize()
    gpu_ms_batch = start_event.elapsed_time(end_event) / REPEATS
    print(f"  GPU inference (batch=16) : {gpu_ms_batch:.3f} ms / batch  "
          f"({gpu_ms_batch/16:.3f} ms / window)")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model_gpu(dummy_b_gpu)
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  GPU peak memory (batch=16): {gpu_mem_mb:.2f} MB")
else:
    print("  GPU not available on this machine.")

print(f"{'='*55}\n")

print(f"  CPU throughput (batch=1)  : {1000/cpu_ms_single:.0f} windows/s")
print(f"  CPU throughput (batch=16) : {1000/(cpu_ms_batch/16):.0f} windows/s")
if torch.cuda.is_available():
    print(f"  GPU throughput (batch=1)  : {1000/gpu_ms_single:.0f} windows/s")
    print(f"  GPU throughput (batch=16) : {1000/(gpu_ms_batch/16):.0f} windows/s")
