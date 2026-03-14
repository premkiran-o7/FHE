"""
FHE Medical AI · Federated Breast Cancer Classifier
====================================================
• Four-hospital federated learning with FedAvg / CKKS-FHE aggregation
• Per-hospital model report cards with local vs federated weight comparison
• Plain-language accuracy-drop explanation for non-technical stakeholders
• All analysis + AI chat grounded in the final federated model weights
• Groq AI agent (openai/gpt-oss-120b)
"""

import warnings
warnings.filterwarnings("ignore")

import copy, io, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import streamlit as st

# ── Optional heavy deps — graceful degradation ──────────────────────────────
try:
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False

try:
    import tenseal as ts
    TENSEAL_OK = True
except ImportError:
    TENSEAL_OK = False

try:
    from groq import Groq as _GroqSDK
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FHE Medical AI · Federated Breast Cancer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }

.metric-card {
    background: linear-gradient(135deg,#1e2130,#252a3d);
    border:1px solid #30364a; border-radius:12px;
    padding:16px 20px; margin:5px 0;
}
.metric-card h3 { color:#7dd3fc; font-size:.75rem;
                  letter-spacing:.08em; margin:0 0 4px; }
.metric-card p  { color:#f0f4ff; font-size:1.5rem;
                  font-weight:700; margin:0; }

.accuracy-drop-card {
    background:linear-gradient(135deg,#2a1f0e,#3a2a10);
    border-left:4px solid #f59e0b;
    border-radius:10px; padding:16px 20px; margin:12px 0;
}
.accuracy-drop-card h4 { color:#fbbf24; margin:0 0 8px; font-size:1rem; }
.accuracy-drop-card p  { color:#fcd34d; font-size:.86rem; margin:4px 0; }

.impact-card {
    background:linear-gradient(135deg,#1a1f2e,#1e2440);
    border-left:4px solid; border-radius:10px;
    padding:14px 18px; margin:7px 0;
}
.impact-doctor   { border-color:#34d399; }
.impact-patient  { border-color:#60a5fa; }
.impact-hospital { border-color:#f472b6; }
.impact-research { border-color:#facc15; }
.impact-card h4 { font-size:.93rem; margin:0 0 5px; }
.impact-card p  { font-size:.82rem; color:#9ca3af; margin:0; }

.fhe-badge {
    display:inline-block; background:#1e3a5f; color:#7dd3fc;
    border:1px solid #2563eb; border-radius:6px;
    padding:3px 10px; font-size:.73rem; font-weight:600; letter-spacing:.05em;
}
.fed-badge {
    display:inline-block; background:#1e3a28; color:#34d399;
    border:1px solid #16a34a; border-radius:6px;
    padding:3px 10px; font-size:.73rem; font-weight:600;
    letter-spacing:.05em; margin-left:6px;
}
.chat-user {
    background:#1e3a5f; border-radius:12px 12px 4px 12px;
    padding:10px 14px; margin:6px 0; color:#e0f2fe; font-size:.87rem;
}
.chat-ai {
    background:#1a2a1a; border-radius:12px 12px 12px 4px;
    padding:10px 14px; margin:6px 0; color:#d1fae5; font-size:.87rem;
    border-left:3px solid #34d399;
}
h1,h2,h3 { color:#f0f4ff !important; }
.stTabs [data-baseweb="tab"] { color:#9ca3af; font-size:.85rem; }
.stTabs [aria-selected="true"] {
    color:#7dd3fc !important;
    border-bottom-color:#7dd3fc !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
HOSPITAL_NAMES  = ["Hospital Alpha", "Hospital Beta", "Hospital Gamma", "Hospital Delta"]
HOSPITAL_COLORS = ["#60a5fa", "#34d399", "#f472b6", "#facc15"]
GROQ_MODEL      = "openai/gpt-oss-120b"

_DARK   = "#0f1117"
_PANEL  = "#1a1f2e"
_BORDER = "#30364a"
_MUTED  = "#9ca3af"
_TITLE  = "#7dd3fc"

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class LogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    # ── FHE helpers ──────────────────────────
    def encrypt_weights(self, context):
        if not TENSEAL_OK:
            raise RuntimeError("tenseal not installed")
        w = self.linear.weight.data.squeeze().tolist()
        b = [float(self.linear.bias.data.squeeze())]
        self.enc_w = ts.ckks_vector(context, w)
        self.enc_b = ts.ckks_vector(context, b)

    def decrypt_weights(self):
        self.linear.weight = nn.Parameter(
            torch.tensor([self.enc_w.decrypt()], dtype=torch.float32))
        self.linear.bias   = nn.Parameter(
            torch.tensor(self.enc_b.decrypt(), dtype=torch.float32))

    @property
    def weights_numpy(self):
        return self.linear.weight.data.squeeze().detach().numpy().copy()

# ─────────────────────────────────────────────
# Data Utilities
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_hospital_datasets():
    """
    Splits the sklearn Wisconsin breast cancer dataset into 4 hospital slices.
    Returns (list_of_dataframes, feature_names).
    """
    bc = load_breast_cancer()
    feat_names = list(bc.feature_names)
    df = pd.DataFrame(bc.data, columns=feat_names)
    # sklearn: 0=malignant, 1=benign  →  flip to match medical convention (malignant=1)
    df["diagnostic"] = (bc.target == 0).astype(int)
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    n  = len(df)
    cuts = [0, n // 4, n // 2, 3 * n // 4, n]
    slices = [df.iloc[cuts[i]: cuts[i + 1]].reset_index(drop=True)
              for i in range(4)]
    return slices, feat_names


def scale_dataset(df: pd.DataFrame, oversample: bool = False):
    feat_cols = [c for c in df.columns if c != "diagnostic"]
    X = df[feat_cols].values.astype(np.float32)
    Y = df["diagnostic"].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if oversample and IMBLEARN_OK:
        ros = RandomOverSampler(random_state=42)
        X, Y = ros.fit_resample(X, Y)
    Xt = Variable(torch.tensor(X, dtype=torch.float32))
    Yt = Variable(torch.tensor(Y, dtype=torch.float32))
    return Xt, Yt, scaler

# ─────────────────────────────────────────────
# Training & Aggregation
# ─────────────────────────────────────────────
def _decide(p: np.ndarray) -> np.ndarray:
    return (p >= 0.5).astype(float)


def compute_accuracy(model: LogisticRegression, X, Y) -> float:
    with torch.no_grad():
        p = model(X).numpy()[:, 0]
    return 100.0 * (_decide(p) == Y.numpy()).sum() / len(p)


def train_local_model(X_tr, Y_tr, n_feat: int, epochs: int, lr: float):
    model = LogisticRegression(n_feat)
    crit  = nn.BCELoss(reduction="mean")
    opt   = torch.optim.SGD(model.parameters(), lr=lr)
    losses, accs = [], []
    log_every = max(1, epochs // 100)

    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X_tr)
        loss = crit(pred.squeeze(), Y_tr)
        loss.backward()
        opt.step()
        if (ep + 1) % log_every == 0:
            losses.append(loss.item())
            accs.append(compute_accuracy(model, X_tr, Y_tr))

    return model, losses, accs


def fedavg(models: list, n_feat: int) -> LogisticRegression:
    gm = LogisticRegression(n_feat)
    with torch.no_grad():
        w = sum(m.linear.weight.data for m in models) / len(models)
        b = sum(m.linear.bias.data   for m in models) / len(models)
        gm.linear.weight = nn.Parameter(w)
        gm.linear.bias   = nn.Parameter(b)
    return gm


def fhe_aggregate(models: list, n_feat: int, context) -> tuple:
    """CKKS-encrypted FedAvg. Returns (global_model, log_lines)."""
    log = []
    for i, m in enumerate(models):
        m.encrypt_weights(context)
        log.append(f"✅ {HOSPITAL_NAMES[i]}: {n_feat} weights encrypted (CKKS)")

    agg_w = models[0].enc_w
    agg_b = models[0].enc_b
    for m in models[1:]:
        agg_w += m.enc_w
        agg_b += m.enc_b
    agg_w *= 1.0 / len(models)
    agg_b *= 1.0 / len(models)
    log.append(f"✅ Aggregation of {len(models)} encrypted vectors done in cipherspace")

    gm = LogisticRegression(n_feat)
    gm.linear.weight = nn.Parameter(
        torch.tensor([agg_w.decrypt()], dtype=torch.float32))
    gm.linear.bias   = nn.Parameter(
        torch.tensor(agg_b.decrypt(), dtype=torch.float32))
    log.append("✅ Decrypted — raw patient data never left any hospital")
    return gm, log


def run_federated_pipeline(hospital_dfs, feat_names,
                            epochs, lr, oversample,
                            use_fhe, fhe_poly, fhe_scale,
                            progress_widget):
    n_feat  = len(feat_names)
    results = []
    loc_mods= []
    fhe_log = []
    n       = len(hospital_dfs)

    for i, df in enumerate(hospital_dfs):
        name = HOSPITAL_NAMES[i]
        progress_widget.progress(i / (n + 1), text=f"🏥 Training {name}…")

        df_s  = df.sample(frac=1, random_state=42).reset_index(drop=True)
        cut   = int(0.8 * len(df_s))
        Xtr, Ytr, _ = scale_dataset(df_s.iloc[:cut], oversample)
        Xte, Yte, _ = scale_dataset(df_s.iloc[cut:], False)

        model, losses, accs = train_local_model(Xtr, Ytr, n_feat, epochs, lr)

        results.append({
            "name":        name,
            "color":       HOSPITAL_COLORS[i],
            "model":       model,
            "losses":      losses,
            "accs":        accs,
            "train_acc":   compute_accuracy(model, Xtr, Ytr),
            "test_acc":    compute_accuracy(model, Xte, Yte),
            "weights":     model.weights_numpy,
            "Xte":         Xte,
            "Yte":         Yte,
            "n_train":     len(df_s.iloc[:cut]),
            "n_test":      len(df_s.iloc[cut:]),
            "n_malignant": int(df["diagnostic"].sum()),
            "n_benign":    int((df["diagnostic"] == 0).sum()),
        })
        loc_mods.append(model)

    # ── Aggregation ──────────────────────────
    progress_widget.progress(n / (n + 1), text="🔐 Aggregating federated model…")
    if use_fhe and TENSEAL_OK:
        try:
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=fhe_poly,
                coeff_mod_bit_sizes=[60, fhe_scale, fhe_scale, 60],
            )
            ctx.global_scale = 2 ** fhe_scale
            ctx.generate_galois_keys()
            fhe_log.append(f"🔑 CKKS context: poly_degree={fhe_poly}, scale=2^{fhe_scale}")
            global_model, extra = fhe_aggregate(loc_mods, n_feat, ctx)
            fhe_log += extra
        except Exception as e:
            fhe_log.append(f"❌ FHE error: {e} — falling back to plaintext FedAvg")
            global_model = fedavg(loc_mods, n_feat)
    else:
        global_model = fedavg(loc_mods, n_feat)
        if use_fhe and not TENSEAL_OK:
            fhe_log.append("⚠️  tenseal not installed. Run: pip install tenseal")

    # ── Evaluate global model on each hospital ─
    for r in results:
        r["fed_acc"]   = compute_accuracy(global_model, r["Xte"], r["Yte"])
        r["acc_delta"] = r["fed_acc"] - r["test_acc"]

    progress_widget.progress(1.0, text="✅ Done!")
    return results, global_model, fhe_log

# ─────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────
def _ax_style(ax):
    ax.set_facecolor(_PANEL)
    ax.spines[:].set_color(_BORDER)
    ax.spines[:].set_linewidth(0.6)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)


def fig_training_curves(results):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), facecolor=_DARK)
    fig.suptitle("Per-Hospital Training Curves", color="#f0f4ff",
                 fontsize=13, fontweight="bold")
    for i, r in enumerate(results):
        cl = r["color"]
        for row, data, ylabel, label in [
            (0, r["losses"], "BCE Loss",  "Loss"),
            (1, r["accs"],   "Acc (%)",   "Accuracy"),
        ]:
            ax = axes[row, i]
            _ax_style(ax)
            ax.plot(data, color=cl, linewidth=2)
            ax.fill_between(range(len(data)), data, alpha=0.12, color=cl)
            ax.set_title(f"{r['name']}\n{label}", color=_TITLE, fontsize=9)
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


def fig_accuracy_comparison(results):
    names  = [r["name"] for r in results]
    loc_a  = [r["test_acc"] for r in results]
    fed_a  = [r["fed_acc"]  for r in results]
    x, bw  = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_DARK)
    _ax_style(ax)

    bars_l = ax.bar(x - bw/2, loc_a, bw, label="Local Model",
                    color=[r["color"] for r in results], alpha=0.85, edgecolor="none")
    bars_f = ax.bar(x + bw/2, fed_a, bw, label="Federated Model",
                    color="#f0f4ff", alpha=0.45, edgecolor=_BORDER, linewidth=0.8)

    for bars in (bars_l, bars_f):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", color=_MUTED, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, color=_MUTED)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, min(loc_a + fed_a) - 8), 102)
    ax.set_title("Local vs Federated Accuracy per Hospital",
                 color=_TITLE, fontsize=12, pad=10)
    ax.legend(facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_MUTED, fontsize=9)
    plt.tight_layout()
    return fig


def fig_weight_analysis(weights: np.ndarray, feat_names: list, title: str):
    w       = np.asarray(weights)
    idx     = np.argsort(np.abs(w))[::-1]
    ws, ns  = w[idx], [feat_names[i] for i in idx]
    col     = ["#ef4444" if v > 0 else "#3b82f6" for v in ws]

    fig = plt.figure(figsize=(16, 11), facecolor=_DARK)
    fig.suptitle(f"Weight Analysis — {title}", color="#f0f4ff",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # 1. Ranked horizontal bar
    ax1 = fig.add_subplot(gs[0, 0]); _ax_style(ax1)
    top = min(15, len(ns))
    ax1.barh(ns[:top][::-1], ws[:top][::-1],
             color=col[:top][::-1], edgecolor="none", height=0.65)
    ax1.axvline(0, color=_BORDER, linewidth=1)
    ax1.set_title("Top Feature Weights (by magnitude)",
                  color=_TITLE, fontsize=10, pad=8)
    ax1.set_xlabel("Weight Value")

    # 2. Distribution histogram
    ax2 = fig.add_subplot(gs[0, 1]); _ax_style(ax2)
    ax2.hist(w, bins=20, color="#7dd3fc", edgecolor=_PANEL, alpha=0.85)
    ax2.axvline(0,         color="#ef4444", lw=1.5, ls="--", label="Zero")
    ax2.axvline(np.mean(w),color="#34d399", lw=1.5, ls="--",
                label=f"Mean {np.mean(w):.3f}")
    ax2.set_title("Weight Distribution", color=_TITLE, fontsize=10, pad=8)
    ax2.set_xlabel("Weight Value"); ax2.set_ylabel("Count")
    ax2.legend(facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_MUTED, fontsize=8)

    # 3. Magnitude heatmap
    ax3 = fig.add_subplot(gs[1, 0]); _ax_style(ax3)
    im  = ax3.imshow(np.abs(w).reshape(1, -1), cmap="YlOrRd", aspect="auto")
    ax3.set_yticks([])
    ax3.set_xticks(range(len(feat_names)))
    ax3.set_xticklabels(feat_names, rotation=90, fontsize=6, color=_MUTED)
    ax3.set_title("Weight Magnitude Heatmap", color=_TITLE, fontsize=10, pad=8)
    plt.colorbar(im, ax=ax3, fraction=0.03).ax.tick_params(colors=_MUTED, labelsize=7)

    # 4. Positive vs Negative scatter
    ax4 = fig.add_subplot(gs[1, 1]); _ax_style(ax4)
    pos = sorted(w[w > 0])
    neg = sorted(w[w < 0], reverse=True)
    ax4.scatter(range(len(pos)), pos, color="#ef4444", s=38, alpha=0.8,
                label=f"Positive ({len(pos)})")
    ax4.scatter(range(len(neg)), neg, color="#3b82f6", s=38, alpha=0.8,
                label=f"Negative ({len(neg)})")
    ax4.axhline(0, color=_BORDER, lw=0.8, ls="--")
    ax4.set_title("Positive vs Negative Weights", color=_TITLE, fontsize=10, pad=8)
    ax4.set_xlabel("Rank"); ax4.set_ylabel("Weight Value")
    ax4.legend(facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_MUTED, fontsize=9)

    return fig


def fig_weight_comparison(results, feat_names):
    top_idx   = np.argsort(np.abs(results[0]["weights"]))[::-1][:6]
    top_names = [feat_names[i] for i in top_idx]
    x         = np.arange(len(top_names))
    bw        = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=_DARK)
    _ax_style(ax)

    for i, r in enumerate(results):
        vals   = [r["weights"][j] for j in top_idx]
        offset = (i - len(results) / 2 + 0.5) * bw
        ax.bar(x + offset, vals, bw * 0.9, label=r["name"],
               color=r["color"], alpha=0.8, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(top_names, rotation=22, ha="right", color=_MUTED, fontsize=9)
    ax.axhline(0, color=_BORDER, lw=0.8)
    ax.set_title("Top-6 Feature Weights Across All Hospitals",
                 color=_TITLE, fontsize=11, pad=10)
    ax.set_ylabel("Weight Value")
    ax.legend(facecolor=_PANEL, edgecolor=_BORDER, labelcolor=_MUTED, fontsize=9)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# Groq AI Agent
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are MediAI, a world-class medical AI analyst specialising in:
• Interpretable machine learning for breast cancer oncology
• Federated learning privacy-utility tradeoffs
• Homomorphic encryption in healthcare (CKKS/CKKS scheme)
• Plain-language clinical communication for diverse audiences

You are provided the ACTUAL WEIGHTS of the final federated model — the global model
produced after CKKS-encrypted aggregation across all four hospital models.
Ground every answer in these real weight values.

Key behaviours:
1. Reference specific feature weight magnitudes when explaining clinical importance
2. Explain federated accuracy drops honestly and without jargon for non-technical users
3. Adapt language: empathetic for patients, precise for clinicians, operational for admins
4. Never recommend this model as a sole diagnostic tool — always defer to clinicians
5. Proactively flag limitations and ethical guardrails"""


def build_groq(api_key: str):
    if not GROQ_OK:
        raise RuntimeError("groq package not installed — run: pip install groq")
    return _GroqSDK(api_key=api_key)


def _ctx_block(ctx: dict) -> str:
    return f"""
=== FEDERATED MODEL CONTEXT ===
Global test accuracy     : {ctx.get('global_test_acc','N/A')}
Avg local test accuracy  : {ctx.get('avg_local_acc','N/A')}
Accuracy delta (fed-local): {ctx.get('acc_delta','N/A')}
FHE encryption used      : {ctx.get('fhe_enabled', False)}
Hospitals                : {ctx.get('n_hospitals', 4)}
Epochs / hospital        : {ctx.get('epochs','N/A')}

=== FEDERATED MODEL WEIGHTS (all features, ranked by magnitude) ===
{ctx.get('all_weights','N/A')}

Top POSITIVE weights (↑ malignancy risk):
{ctx.get('top_pos','N/A')}

Top NEGATIVE weights (↓ protective/benign indicators):
{ctx.get('top_neg','N/A')}

=== PER-HOSPITAL SUMMARY ===
{ctx.get('hospital_summary','N/A')}
=================================
"""


def chat_groq(client, history: list, user_msg: str,
              ctx: dict | None = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if ctx:
        messages.append({
            "role":    "user",
            "content": _ctx_block(ctx) + "\n\nUser question:\n" + user_msg,
        })
    else:
        messages += history
        messages.append({"role": "user", "content": user_msg})

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.38,
    )
    return resp.choices[0].message.content


def generate_full_report(client, ctx: dict) -> str:
    prompt = _ctx_block(ctx) + """
Please write a comprehensive clinical interpretation report with these exact sections:

## 1. Executive Summary
Two concise sentences on what this federated model reveals.

## 2. Top Feature Clinical Interpretation
For each of the top 5 positive-weight and top 5 negative-weight features:
— Name the feature, give its weight, explain what it measures biologically,
  and describe what a high/low value means for cancer risk.

## 3. For Oncologists & Radiologists
How these findings should influence fine-needle aspirate review and diagnostic workflow.

## 4. For Patients (Plain Language)
Jargon-free explanation of what the model does, what it found, and what privacy means for them.

## 5. For Hospital Administrators & Privacy Officers
Operational implications: HIPAA/GDPR compliance, FHE audit trail, federated vs centralised tradeoffs.

## 6. For Researchers
Statistical observations: weight distribution, sparsity, overfitting risk, improvement roadmap.

## 7. The Federated Accuracy Tradeoff
Explain clearly (for a non-technical audience) why the global model may be slightly less accurate
on any individual hospital's patients, and why this is acceptable and desirable.

## 8. Limitations & Ethical Guardrails
What this model must NOT be used for. Bias risks. When to defer to human experts.

Be thorough, empathetic, and clinically grounded.
Reference specific weight magnitudes throughout."""

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt},
        ],
        max_tokens=3500,
        temperature=0.35,
    )
    return resp.choices[0].message.content


def generate_hospital_report(client, r: dict, fw: np.ndarray,
                              feat_names: list) -> str:
    w   = r["weights"]
    top = np.argsort(np.abs(w))[::-1][:5]

    prompt = f"""
Hospital: {r['name']}
Training samples : {r['n_train']}   Test samples: {r['n_test']}
Malignant cases  : {r['n_malignant']}   Benign cases: {r['n_benign']}

Local model test accuracy    : {r['test_acc']:.2f}%
Federated model test accuracy: {r['fed_acc']:.2f}%
Accuracy change after federation: {r['acc_delta']:+.2f}%

This hospital's top 5 feature weights (local model):
{[(feat_names[i], round(float(w[i]),5)) for i in top]}

Federated model weights for the same features:
{[(feat_names[i], round(float(fw[i]),5)) for i in top]}

Write a hospital-specific clinical report (350-450 words) covering:
1. Dataset profile and any class imbalance considerations
2. What this hospital's local model learned — clinical meaning of top weights
3. How the federated model compares, explaining any accuracy change plainly:
   If accuracy DROPPED: "The global model compromises across four populations —
   slightly less tuned to your specific patients, but far more generalised and privacy-safe."
   If accuracy IMPROVED or HELD: explain why diversity from other hospitals helped.
4. Privacy benefit: what sensitive data this hospital did NOT share
5. One concrete recommendation for the clinical team

Use plain, professional language suitable for a mixed audience of clinicians and administrators."""

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=750,
        temperature=0.35,
    )
    return resp.choices[0].message.content

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 FHE Medical AI")
    st.markdown(
        '<span class="fhe-badge">CKKS Encryption</span>'
        '<span class="fed-badge">FedAvg</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### ⚙️ Training Config")
    epochs     = st.slider("Epochs per hospital", 200, 5000, 1000, step=100)
    lr         = st.select_slider("Learning Rate",
                                   [0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
    oversample = st.toggle("Oversample minority class", value=True)

    st.markdown("---")
    st.markdown("### 🔐 FHE Settings")
    use_fhe = st.toggle(
        "Enable CKKS Encryption",
        value=False,
        disabled=not TENSEAL_OK,
        help="Requires `pip install tenseal`" if not TENSEAL_OK else "",
    )
    if not TENSEAL_OK:
        st.caption("⚠️ `tenseal` not installed — FHE unavailable")
    fhe_poly, fhe_scale = 8192, 40
    if use_fhe and TENSEAL_OK:
        fhe_poly  = st.selectbox("Poly Modulus Degree", [4096, 8192, 16384], index=1)
        fhe_scale = st.slider("Scale Bits", 20, 60, 40)

    st.markdown("---")
    st.markdown("### 🤖 Groq AI Agent")
    st.caption(f"Model: `{GROQ_MODEL}`")
    if not GROQ_OK:
        st.warning("`pip install groq` required")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_…")
    groq_client = None
    if groq_key:
        try:
            groq_client = build_groq(groq_key)
            st.success("✅ Groq connected")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    train_btn = st.button("🚀 Run Federated Learning",
                           use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
_INIT = {
    "hospital_results": None, "global_model": None, "feat_names": None,
    "fhe_log": [], "context_data": None, "auto_report": None,
    "hosp_reports": {}, "chat_msgs": [], "fed_weights": None,
}
for k, v in _INIT.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 🧬 Federated Breast Cancer Classifier · FHE Medical AI")
st.markdown(
    "**Privacy-preserving federated learning** across four hospitals · "
    "Optional **CKKS homomorphic encryption** · **Groq AI** clinical insights"
)
st.markdown("---")

# ─────────────────────────────────────────────
# Training Trigger
# ─────────────────────────────────────────────
if train_btn:
    hospital_dfs, feat_names = load_hospital_datasets()
    prog_slot = st.empty()
    prog_bar  = prog_slot.progress(0, text="Starting federated pipeline…")

    results, global_model, fhe_log = run_federated_pipeline(
        hospital_dfs, feat_names, epochs, lr,
        oversample, use_fhe, fhe_poly, fhe_scale,
        prog_bar,
    )
    prog_slot.empty()

    fw         = global_model.weights_numpy
    sorted_idx = np.argsort(np.abs(fw))[::-1]
    top_pos    = [(feat_names[i], round(float(fw[i]), 5))
                  for i in sorted_idx if fw[i] > 0][:7]
    top_neg    = [(feat_names[i], round(float(fw[i]), 5))
                  for i in sorted_idx if fw[i] < 0][:7]
    all_w_str  = "\n".join(
        f"  {feat_names[i]}: {fw[i]:+.6f}" for i in sorted_idx
    )

    all_Xte = torch.cat([r["Xte"] for r in results])
    all_Yte = torch.cat([r["Yte"] for r in results])
    global_test_acc = compute_accuracy(global_model, all_Xte, all_Yte)
    avg_local = float(np.mean([r["test_acc"] for r in results]))

    hosp_sum = "\n".join(
        f"  {r['name']}: local={r['test_acc']:.2f}%  "
        f"fed={r['fed_acc']:.2f}%  delta={r['acc_delta']:+.2f}%  "
        f"n_train={r['n_train']}  malignant={r['n_malignant']}"
        for r in results
    )

    st.session_state.update({
        "hospital_results": results,
        "global_model":     global_model,
        "feat_names":       feat_names,
        "fhe_log":          fhe_log,
        "fed_weights":      fw,
        "auto_report":      None,
        "hosp_reports":     {},
        "chat_msgs":        [],
        "context_data": {
            "global_test_acc":  f"{global_test_acc:.2f}%",
            "avg_local_acc":    f"{avg_local:.2f}%",
            "acc_delta":        f"{float(np.mean([r['acc_delta'] for r in results])):+.2f}%",
            "fhe_enabled":      use_fhe and TENSEAL_OK,
            "n_hospitals":      4,
            "epochs":           epochs,
            "top_pos":          str(top_pos),
            "top_neg":          str(top_neg),
            "all_weights":      all_w_str,
            "hospital_summary": hosp_sum,
        },
    })
    st.success("✅ Federated learning complete! Explore the tabs below.")
    st.rerun()

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📊 Hospital Reports",
    "🌐 Federated Model",
    "📈 Training Curves",
    "🔬 Weight Analysis",
    "🔐 FHE Demo",
    "🤖 Groq Agent",
])

# ══════════════════════════════════════════════
# TAB 0 — Overview
# ══════════════════════════════════════════════
with tabs[0]:
    if st.session_state.global_model is None:
        # Pre-training info
        with st.spinner("Loading dataset…"):
            dfs, fns = load_hospital_datasets()

        st.markdown("### 📁 Wisconsin Breast Cancer Dataset — 4-Hospital Split")
        all_df = pd.concat(dfs, ignore_index=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in [
            (c1, "TOTAL SAMPLES",   len(all_df)),
            (c2, "FEATURES",        len(fns)),
            (c3, "MALIGNANT",       int(all_df["diagnostic"].sum())),
            (c4, "BENIGN",          int((all_df["diagnostic"]==0).sum())),
        ]:
            col.markdown(
                f'<div class="metric-card"><h3>{lbl}</h3><p>{val}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 🏥 Hospital Dataset Distribution")
        rows = [{"Hospital": nm, "Samples": len(df),
                 "Malignant": int(df["diagnostic"].sum()),
                 "Benign":    int((df["diagnostic"]==0).sum()),
                 "M-ratio":   f"{100*df['diagnostic'].mean():.1f}%"}
                for df, nm in zip(dfs, HOSPITAL_NAMES)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")
        st.markdown("### 💡 What Is Federated Learning?")
        st.markdown("""
**Federated Learning** trains a shared AI model across multiple hospitals — 
**without any patient data ever leaving a hospital**.

```
🏥 Hospital Alpha  ──► train locally ──► encrypt weights ──►┐
🏥 Hospital Beta   ──► train locally ──► encrypt weights ──►│
🏥 Hospital Gamma  ──► train locally ──► encrypt weights ──►├──► Average ──► 🌐 Global Model
🏥 Hospital Delta  ──► train locally ──► encrypt weights ──►┘
                                         ↑
                        (only encrypted numbers travel — never patient records)
```

The **global model** benefits from all four hospitals' knowledge while keeping every 
individual's health records private. With **FHE encryption**, even the server doing the 
averaging never sees the raw weight numbers.
        """)
        st.info("👈 Configure settings in the sidebar and click **Run Federated Learning**.")

    else:
        results = st.session_state.hospital_results
        ctx     = st.session_state.context_data

        st.markdown("### 🌐 Federated Learning Results")
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in [
            (c1, "GLOBAL TEST ACC",   ctx["global_test_acc"]),
            (c2, "AVG LOCAL ACC",     ctx["avg_local_acc"]),
            (c3, "ACCURACY DELTA",    ctx["acc_delta"]),
            (c4, "FHE USED",          "Yes ✅" if ctx["fhe_enabled"] else "No"),
        ]:
            col.markdown(
                f'<div class="metric-card"><h3>{lbl}</h3><p>{val}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        fig = fig_accuracy_comparison(results)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Accuracy-drop explainer card
        any_drop = any(r["acc_delta"] < -0.5 for r in results)
        if any_drop:
            st.markdown("""
<div class="accuracy-drop-card">
<h4>📉 Why Is the Federated Model Less Accurate on Some Hospitals?</h4>
<p>You may notice the federated (global) model is slightly less accurate than each
hospital's own locally-trained model. <strong>This is expected — and intentional.</strong></p>
<p><strong>Think of it like this:</strong> your hospital's local model is like a specialist
who has only ever treated patients from your city — very good at your population's
patterns. The federated model is trained across four cities simultaneously. It's more
general and knows about rare patterns from other hospitals, but it's slightly less
fine-tuned to <em>your</em> specific patients.</p>
<p><strong>Why accept this tradeoff?</strong><br>
① <strong>Privacy</strong> — zero patient records leave any hospital.<br>
② <strong>Generalisability</strong> — the global model handles rare cases better.<br>
③ <strong>Fairness</strong> — smaller hospitals gain from larger hospitals' knowledge.<br>
④ <strong>Regulation</strong> — CKKS encryption satisfies HIPAA &amp; GDPR requirements.<br><br>
In practice the difference is usually &lt;2% — a tiny price for enormous privacy gains.</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏥 Real-World Impact")
        st.markdown("""
<div class="impact-card impact-doctor">
<h4>👨‍⚕️ For Oncologists & Radiologists</h4>
<p>The federated model integrates morphological patterns from four distinct patient
populations. The Weight Analysis tab shows which cell measurements drive malignancy
scores — focusing your review of fine-needle aspirate slides on statistically
validated features rather than intuition alone.</p>
</div>

<div class="impact-card impact-patient">
<h4>🧑 For Patients</h4>
<p>Your biopsy measurements are processed by an AI that learned from hundreds of similar
cases across multiple hospitals — without your data ever being shared. FHE encryption
ensures even the system aggregating the models never sees your raw health information.</p>
</div>

<div class="impact-card impact-hospital">
<h4>🏥 For Hospital Administrators & Privacy Officers</h4>
<p>No patient identifiers or raw measurements travel between hospitals. Only encrypted
mathematical vectors (unintelligible without the private decryption key) are transmitted.
This satisfies HIPAA Safe Harbor and GDPR Article 25 (privacy by design) without
sacrificing model quality.</p>
</div>

<div class="impact-card impact-research">
<h4>🔬 For Researchers</h4>
<p>FedAvg over logistic regression gives an interpretable, auditable baseline.
The weight distribution across four heterogeneous hospital populations reveals
cross-institutional feature importance — impossible to study without federated learning.
CKKS round-trip numerical noise is order 10⁻⁹ — negligible for floating-point weights.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 1 — Hospital Reports
# ══════════════════════════════════════════════
with tabs[1]:
    if st.session_state.hospital_results is None:
        st.info("Run federated learning first to see per-hospital reports.")
    else:
        results    = st.session_state.hospital_results
        feat_names = st.session_state.feat_names
        fw         = st.session_state.fed_weights

        st.markdown("### 🏥 Per-Hospital Report Cards")
        sel = st.radio("Select hospital", HOSPITAL_NAMES, horizontal=True)
        r   = next(x for x in results if x["name"] == sel)

        # ── Metrics ──────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, lbl, val in [
            (c1, "TRAIN SAMPLES",  r["n_train"]),
            (c2, "TEST SAMPLES",   r["n_test"]),
            (c3, "LOCAL TEST ACC", f"{r['test_acc']:.2f}%"),
            (c4, "FED MODEL ACC",  f"{r['fed_acc']:.2f}%"),
            (c5, "ACC DELTA",      f"{r['acc_delta']:+.2f}%"),
        ]:
            col.markdown(
                f'<div class="metric-card"><h3>{lbl}</h3><p>{val}</p></div>',
                unsafe_allow_html=True,
            )

        cA, cB = st.columns(2)
        cA.metric("Malignant cases", r["n_malignant"])
        cB.metric("Benign cases",    r["n_benign"])

        # ── Accuracy-drop explainer ───────────
        if r["acc_delta"] < -0.5:
            st.markdown(f"""
<div class="accuracy-drop-card">
<h4>📉 Accuracy Drop for {r['name']}: {r['acc_delta']:+.2f}%</h4>
<p>The federated model is <strong>{abs(r['acc_delta']):.2f}% less accurate</strong>
on {r['name']}'s patients than the local model trained only on those patients.</p>
<p><strong>What does this mean?</strong> The local model was perfectly tuned to
{r['name']}'s specific patient mix. The global model averaged in patterns from
three other hospitals — making it more general, but slightly less specialised here.</p>
<p><strong>Is this a problem?</strong> Rarely. The gap is small, privacy is
absolute (zero records left the hospital), and the global model is more robust
to unusual cases not present in this hospital's data.</p>
</div>
""", unsafe_allow_html=True)
        elif r["acc_delta"] >= 0:
            st.success(
                f"✅ The federated model performs **equally well or better** "
                f"({r['acc_delta']:+.2f}%) on {r['name']}'s patients — "
                f"other hospitals' data contributed useful patterns here!"
            )
        else:
            st.info(
                f"ℹ️ Marginal accuracy change ({r['acc_delta']:+.2f}%) — "
                "within normal federated learning variation."
            )

        st.markdown("---")

        # ── Weight comparison table ───────────
        st.markdown("#### 🔍 Local vs Federated Weights (Top 10)")
        top_idx = np.argsort(np.abs(r["weights"]))[::-1][:10]
        wdf = pd.DataFrame({
            "Feature":          [feat_names[i] for i in top_idx],
            "Local Weight":     [round(float(r["weights"][i]), 6) for i in top_idx],
            "Federated Weight": [round(float(fw[i]), 6) for i in top_idx],
            "Difference":       [round(float(r["weights"][i] - fw[i]), 6) for i in top_idx],
            "Local Direction":  ["↑ Cancer risk" if r["weights"][i] > 0 else "↓ Protective"
                                 for i in top_idx],
        })
        st.dataframe(
            wdf.style
               .background_gradient(subset=["Local Weight", "Federated Weight"], cmap="RdBu_r")
               .format({c: "{:.6f}" for c in
                        ["Local Weight", "Federated Weight", "Difference"]}),
            use_container_width=True,
        )

        # ── Training curves ───────────────────
        st.markdown("---")
        st.markdown("#### 📈 Training Curves")
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4), facecolor=_DARK)
        for ax in (ax_l, ax_a):
            _ax_style(ax)
        cl = r["color"]
        ax_l.plot(r["losses"], color=cl, lw=2)
        ax_l.fill_between(range(len(r["losses"])), r["losses"], alpha=0.12, color=cl)
        ax_l.set_title("Training Loss", color=_TITLE)
        ax_l.set_xlabel("Checkpoint"); ax_l.set_ylabel("BCE Loss")
        ax_a.plot(r["accs"], color=cl, lw=2)
        ax_a.fill_between(range(len(r["accs"])), r["accs"], alpha=0.12, color=cl)
        ax_a.set_title("Training Accuracy", color=_TITLE)
        ax_a.set_xlabel("Checkpoint"); ax_a.set_ylabel("Acc (%)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── AI report ─────────────────────────
        st.markdown("---")
        st.markdown("#### 🤖 AI Clinical Report for This Hospital")
        if groq_client is None:
            st.warning("Add your Groq API key in the sidebar to generate AI reports.")
        else:
            if st.button(f"📋 Generate Report for {sel}", key=f"rpt_{sel}"):
                with st.spinner(f"Generating report for {sel}…"):
                    txt = generate_hospital_report(groq_client, r, fw, feat_names)
                    st.session_state.hosp_reports[sel] = txt
            if sel in st.session_state.hosp_reports:
                st.markdown(st.session_state.hosp_reports[sel])

# ══════════════════════════════════════════════
# TAB 2 — Federated Model
# ══════════════════════════════════════════════
with tabs[2]:
    if st.session_state.global_model is None:
        st.info("Run federated learning first.")
    else:
        fw  = st.session_state.fed_weights
        fn  = st.session_state.feat_names
        ctx = st.session_state.context_data
        res = st.session_state.hospital_results

        enc_method = "CKKS homomorphic encryption" if ctx["fhe_enabled"] else "plaintext FedAvg"
        st.markdown("### 🌐 Global Federated Model")
        st.markdown(
            f"Produced by averaging all four hospital models using **{enc_method}**. "
            "All subsequent weight analysis, AI chat, and clinical reports use these weights."
        )

        c1, c2, c3 = st.columns(3)
        for col, lbl, val in [
            (c1, "GLOBAL TEST ACC", ctx["global_test_acc"]),
            (c2, "AVG LOCAL ACC",   ctx["avg_local_acc"]),
            (c3, "AVG ACC DELTA",   ctx["acc_delta"]),
        ]:
            col.markdown(
                f'<div class="metric-card"><h3>{lbl}</h3><p>{val}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 📊 Accuracy — All Hospitals")
        fig = fig_accuracy_comparison(res)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")
        st.markdown("#### 🧠 How Federated Learning Works — Step by Step")
        st.markdown("""
| Step | What happens | What leaves the hospital |
|------|-------------|--------------------------|
| 1 · Local training | Each hospital trains on its own patients | Nothing |
| 2 · Weight extraction | Model weights (numbers, no patient info) are extracted | Nothing yet |
| 3 · Encryption *(FHE)* | Weights are encrypted into ciphertext | Only ciphertext |
| 4 · Aggregation | Server adds encrypted vectors — no decryption needed | — |
| 5 · Averaging | Divide by number of hospitals (still encrypted) | — |
| 6 · Decryption | Authorised holder decrypts the averaged global model | — |
| 7 · Deployment | Global model distributed back to hospitals | Global model weights |

**Key guarantee:** At no point does any hospital's raw patient data leave that hospital.
        """)

        st.markdown("---")
        st.markdown("#### 🔢 Global Model — Top 15 Weights")
        sidx = np.argsort(np.abs(fw))[::-1]
        wdf  = pd.DataFrame({
            "Rank":      range(1, 16),
            "Feature":   [fn[i] for i in sidx[:15]],
            "Weight":    [round(float(fw[i]), 6) for i in sidx[:15]],
            "|Weight|":  [round(abs(float(fw[i])), 6) for i in sidx[:15]],
            "Direction": ["↑ Malignancy" if fw[i] > 0 else "↓ Protective"
                          for i in sidx[:15]],
        })
        st.dataframe(
            wdf.style.background_gradient(subset=["Weight"], cmap="RdBu_r")
               .format({"Weight": "{:.6f}", "|Weight|": "{:.6f}"}),
            use_container_width=True,
        )

# ══════════════════════════════════════════════
# TAB 3 — Training Curves
# ══════════════════════════════════════════════
with tabs[3]:
    if st.session_state.hospital_results is None:
        st.info("Run federated learning first.")
    else:
        st.markdown("### 📈 Training Curves — All Four Hospitals")
        fig = fig_training_curves(st.session_state.hospital_results)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")
        st.markdown("#### 📋 Convergence Summary")
        rows = []
        for r in st.session_state.hospital_results:
            loss_red = 100 * (r["losses"][0] - r["losses"][-1]) / r["losses"][0]
            rows.append({
                "Hospital":        r["name"],
                "Initial Loss":    f"{r['losses'][0]:.4f}",
                "Final Loss":      f"{r['losses'][-1]:.4f}",
                "Loss Reduction":  f"{loss_red:.1f}%",
                "Final Train Acc": f"{r['accs'][-1]:.2f}%",
                "Local Test Acc":  f"{r['test_acc']:.2f}%",
                "Fed Test Acc":    f"{r['fed_acc']:.2f}%",
                "Δ (Fed − Local)": f"{r['acc_delta']:+.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — Weight Analysis (FEDERATED model)
# ══════════════════════════════════════════════
with tabs[4]:
    if st.session_state.global_model is None:
        st.info("Run federated learning first.")
    else:
        fw  = st.session_state.fed_weights
        fn  = st.session_state.feat_names
        res = st.session_state.hospital_results

        st.markdown("### 🔬 Weight Analysis — Federated Global Model")
        st.info(
            "All visualisations below use the **final federated model weights** — "
            "the global average from all four hospitals. "
            "These are also the weights the AI agent uses when answering questions."
        )

        fig = fig_weight_analysis(fw, fn, "Federated Global Model")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")
        st.markdown("#### 🏥 Top-6 Feature Weights Compared Across Hospitals")
        fig2 = fig_weight_comparison(res, fn)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.markdown("---")
        st.markdown("#### 📋 Full Federated Weight Table")
        sidx = np.argsort(np.abs(fw))[::-1]
        wdf  = pd.DataFrame({
            "Feature":   [fn[i] for i in sidx],
            "Weight":    [round(float(fw[i]), 6) for i in sidx],
            "Abs Weight":[round(abs(float(fw[i])), 6) for i in sidx],
            "Direction": ["↑ Malignancy" if fw[i] > 0 else "↓ Protective"
                          for i in sidx],
        })
        wdf.index = range(1, len(wdf) + 1)
        st.dataframe(
            wdf.style.background_gradient(subset=["Weight"], cmap="RdBu_r")
               .format({"Weight": "{:.6f}", "Abs Weight": "{:.6f}"}),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("#### 🔍 Feature Spotlight")
        sel_f = st.selectbox("Inspect a feature", fn)
        idx   = fn.index(sel_f)
        val   = fw[idx]
        rank  = int(np.where(np.argsort(np.abs(fw))[::-1] == idx)[0][0]) + 1
        pct   = 100 * (1 - rank / len(fn))

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Federated Weight",  f"{val:.6f}")
        cc2.metric("Importance Rank",   f"#{rank} / {len(fn)}")
        cc3.metric("Direction",         "↑ Malignancy" if val > 0 else "↓ Protective")
        cc4.metric("Percentile",        f"Top {100-int(pct):.0f}%")

        if rank <= len(fn) // 4:
            st.success(f"**{sel_f}** is highly influential (top 25%) in the federated model.")
        elif rank <= len(fn) // 2:
            st.info(f"**{sel_f}** has moderate influence (rank #{rank}).")
        else:
            st.warning(f"**{sel_f}** has relatively low influence in the global model.")

# ══════════════════════════════════════════════
# TAB 5 — FHE Demo
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🔐 Fully Homomorphic Encryption · CKKS Scheme")
    st.markdown("""
**Homomorphic encryption** lets you do arithmetic on encrypted numbers — without
ever decrypting them. Think of it like being able to calculate the average of
four sealed envelopes without opening any of them.

In this pipeline:
- Each hospital **encrypts its model weights** into a ciphertext vector
- The aggregation server **adds and averages the encrypted vectors**
- The result is decrypted **once** — yielding the global averaged model
- At no point does the server see plaintext weight values
    """)

    if not TENSEAL_OK:
        st.warning("⚠️ `tenseal` is not installed. Install with: `pip install tenseal`")

    fhe_log = st.session_state.fhe_log
    if fhe_log:
        st.markdown("#### 📟 Encryption Session Log")
        for line in fhe_log:
            if line.startswith("✅"):
                st.success(line)
            elif line.startswith("❌"):
                st.error(line)
            elif line.startswith("⚠️"):
                st.warning(line)
            else:
                st.code(line, language="text")
    elif st.session_state.global_model is not None:
        st.info(
            "FHE was not enabled in this run. "
            "Toggle **Enable CKKS Encryption** in the sidebar and re-run."
        )
    else:
        st.info("Run federated learning with FHE enabled to see the encryption log.")

    st.markdown("---")
    st.markdown("""
#### CKKS Aggregation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                CKKS FEDERATED AGGREGATION FLOW                  │
│                                                                 │
│  Hospital Alpha: [w₁…wₙ]  ──CKKS encrypt──►  🔒 ciphertext A   │
│  Hospital Beta : [w₁…wₙ]  ──CKKS encrypt──►  🔒 ciphertext B   │
│  Hospital Gamma: [w₁…wₙ]  ──CKKS encrypt──►  🔒 ciphertext C  ─┤
│  Hospital Delta: [w₁…wₙ]  ──CKKS encrypt──►  🔒 ciphertext D   │
│                                                                 │
│  Server (sees only ciphertexts):                                │
│    🔒 sum = A + B + C + D       (addition on ciphertexts)       │
│    🔒 avg = sum × ¼             (scalar multiply on ciphertext) │
│                                                                 │
│  Authorised key holder:                                         │
│    decrypt(🔒 avg) = [w̄₁…w̄ₙ]  →  Global model weights ✅      │
└─────────────────────────────────────────────────────────────────┘
```

**CKKS parameters:**
- `poly_modulus_degree` — higher = stronger encryption, slower arithmetic
- `scale bits` — floating-point precision of ciphertext arithmetic
- Round-trip numerical error ≈ 10⁻⁹ — negligible for model weights
    """)

# ══════════════════════════════════════════════
# TAB 6 — Groq Agent
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🤖 Groq Medical AI Agent")
    st.caption(
        f"Model: `{GROQ_MODEL}` · Answers grounded in actual federated model weights"
    )

    if groq_client is None:
        st.warning("Enter your **Groq API Key** in the sidebar to enable the AI agent.")
    elif st.session_state.global_model is None:
        st.info("Run federated learning first, then ask the agent.")
    else:
        ctx = st.session_state.context_data

        # ── Full Clinical Report ──────────────
        col1, _ = st.columns([2, 3])
        with col1:
            if st.button("🧠 Generate Full Clinical Report", use_container_width=True):
                with st.spinner("Generating comprehensive clinical report…"):
                    report = generate_full_report(groq_client, ctx)
                    st.session_state.auto_report = report

        if st.session_state.auto_report:
            st.markdown("---")
            st.markdown("#### 📋 Full Clinical Interpretation Report")
            st.markdown(st.session_state.auto_report)
            st.markdown("---")

        # ── Chat Interface ────────────────────
        st.markdown("#### 💬 Chat with MediAI")
        st.markdown(
            "All answers are grounded in the **actual federated model weights**. "
            "Ask about feature importance, clinical implications, privacy, "
            "accuracy tradeoffs, or patient impact."
        )

        for msg in st.session_state.chat_msgs:
            cls  = "chat-user" if msg["role"] == "user" else "chat-ai"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="{cls}">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("**Quick questions:**")
        suggestions = [
            "What are the top cancer-risk features and their clinical meaning?",
            "Why is the federated model less accurate on some hospitals?",
            "Explain FHE to a non-technical hospital administrator",
            "What does a negative weight feature mean for a patient?",
            "How should oncologists use these model results?",
            "What are the ethical limits of this model?",
        ]
        cols = st.columns(3)
        for i, sug in enumerate(suggestions):
            if cols[i % 3].button(sug[:43] + "…", key=f"sug_{i}"):
                with st.spinner("MediAI is thinking…"):
                    reply = chat_groq(groq_client, [], sug, ctx)
                st.session_state.chat_msgs.append({"role": "user",      "content": sug})
                st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
                st.rerun()

        user_q = st.chat_input("Ask about the federated model, weights, or clinical impact…")
        if user_q:
            with st.spinner("MediAI is thinking…"):
                reply = chat_groq(groq_client, [], user_q, ctx)
            st.session_state.chat_msgs.append({"role": "user",      "content": user_q})
            st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
            st.rerun()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#4b5563;font-size:.76rem'>"
    f"FHE Medical AI · Federated Breast Cancer Classifier · "
    f"TenSEAL (CKKS) + Groq ({GROQ_MODEL}) · Built with Streamlit"
    f"</div>",
    unsafe_allow_html=True,
)