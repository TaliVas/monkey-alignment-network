"""
HFS decoding: classify single-trial reaching trajectories as Control vs. HFS
(high-frequency cerebellar-block) using a bidirectional LSTM.

Reproduces the within-animal decoding results (Figs. 4-5 in the paper).

Usage:
    python src/hfs_decoding_train_eval.py --data data/Processed_Trajectories_Nana_-300_+500.pkl
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------- config -----------------------------
WIN_START, WIN_END, ONSET = 200, 600, 300   # -100 ms to +300 ms around movement onset
MIN_SAMPLES, MAX_SAMPLES = 35, 50           # keep well-formed trajectories
FEATURE_COLS = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
SEED = 42


# ----------------------------- data -------------------------------
def load_and_window(path):
    df_full = pd.read_pickle(path)
    rows = []
    for tid in df_full['id'].unique():
        traj = df_full[df_full['id'] == tid].sort_values('time_milisecond')
        w = traj[(traj['adjusted_time'] >= WIN_START) & (traj['adjusted_time'] <= WIN_END)]
        if len(w) < 10:
            continue
        rows.append({
            'id': tid, 'type': traj['type'].iloc[0], 'target': traj['id_target'].iloc[0],
            'seq_length': len(w),
            'pos_x': w['centered_x'].values, 'pos_y': w['centered_y'].values, 'pos_z': w['centered_z'].values,
            'vel_x': w['dx'].values, 'vel_y': w['dy'].values, 'vel_z': w['dz'].values,
        })
    df = pd.DataFrame(rows)
    df = df[(df['seq_length'] >= MIN_SAMPLES) & (df['seq_length'] <= MAX_SAMPLES)].reset_index(drop=True)
    return df


def balance_per_target(df):
    parts = []
    for tid in sorted(df['target'].unique()):
        t = df[df['target'] == tid]
        ctrl, hfs = t[t['type'] == 'Control'], t[t['type'] == 'HFS']
        n = min(len(ctrl), len(hfs))
        if n == 0:
            continue
        parts.append(pd.concat([ctrl.sample(n, random_state=SEED),
                                hfs.sample(n, random_state=SEED)]))
    return pd.concat(parts).reset_index(drop=True)


def pad_sequences(df, cols, max_len):
    X = np.zeros((len(df), max_len, len(cols)), dtype=np.float32)
    lengths = np.zeros(len(df), dtype=np.int64)
    for i, (_, row) in enumerate(df.iterrows()):
        L = row['seq_length']; lengths[i] = L
        for j, c in enumerate(cols):
            X[i, :L, j] = row[c]
    return X, lengths


class TrajectoryDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X, self.y, self.lengths = torch.FloatTensor(X), torch.FloatTensor(y), torch.LongTensor(lengths)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.lengths[i]


# ----------------------------- model ------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size * 2, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(out, batch_first=True)
        pooled = self.pool(unpacked.permute(0, 2, 1)).squeeze(-1)
        return self.fc2(self.relu(self.fc1(pooled)))


def train_and_evaluate(Xtr, ytr, ltr, Xte, yte, lte, input_size, epochs=50, lr=1e-3):
    tr = DataLoader(TrajectoryDataset(Xtr, ytr, ltr), batch_size=32, shuffle=True)
    te = DataLoader(TrajectoryDataset(Xte, yte, lte), batch_size=32)
    model = LSTMClassifier(input_size)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for seqs, labels, lens in tr:
            opt.zero_grad()
            loss = crit(model(seqs, lens).squeeze(), labels)
            loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seqs, labels, lens in te:
            p = (torch.sigmoid(model(seqs, lens).squeeze()) > 0.5).int().numpy()
            preds.extend(np.atleast_1d(p)); trues.extend(labels.int().numpy())
    return np.array(preds), np.array(trues)


# ----------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Processed trajectory pickle (e.g. Nana or Thina)')
    ap.add_argument('--epochs', type=int, default=50)
    args = ap.parse_args()

    df = load_and_window(args.data)
    df = balance_per_target(df)
    max_len = df['seq_length'].max()
    X, lengths = pad_sequences(df, FEATURE_COLS, max_len)
    y = (df['type'] == 'HFS').astype(int).values
    targets = df['target'].values

    strat = df['type'].astype(str) + '_' + df['target'].astype(str)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED, stratify=strat)

    preds, trues = train_and_evaluate(X[tr], y[tr], lengths[tr], X[te], y[te], lengths[te],
                                      input_size=len(FEATURE_COLS), epochs=args.epochs)

    print(f"\nOverall accuracy: {accuracy_score(trues, preds):.3f}")
    te_targets = targets[te]
    for tid in sorted(np.unique(te_targets)):
        m = te_targets == tid
        if m.sum():
            print(f"  Target {int(tid)}: {accuracy_score(trues[m], preds[m]):.3f}  (n={m.sum()})")


if __name__ == '__main__':
    main()
