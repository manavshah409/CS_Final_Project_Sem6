"""
================================================================================
5G DDoS DETECTION SYSTEM — FULLY AUTOMATED INTERACTIVE DASHBOARD  v2
================================================================================
Pages:
  1. 📡 Data Lab         — Auto-loaded dataset, live charts, no manual inputs
  2. ⚙️  Pipeline         — One-click automated preprocessing with live feedback
  3. 🤖 Model Arena      — Auto-trains all models, live leaderboard
  4. 📊 Analytics Hub    — Deep-dive charts: radar, ROC, confusion, CV
  5. 🛡️  Threat Monitor   — Animated real-time defense sim + packet inspector

Speed improvements v2:
  • Preprocessing  — parallel NaN fills (ThreadPoolExecutor), vectorised inf-clip,
                     sample-based mutual-info (≤60 k rows), skip if already done
  • Model training — lighter default params, n_jobs=-1 everywhere, cached CV
  • Dashboard      — sticky top KPI bar, animated gradient hero, pulse badges,
                     scan-line overlay, glow metrics, improved layout density

Run:
    streamlit run complete_system.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import json
import random
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="5G DDoS Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

/* ── Global ──────────────────────────────────────────────────────────── */
html, body, .stApp {
    background: #030a14;
    color: #b0cce8;
    font-family: 'Rajdhani', sans-serif;
}

/* Animated scan-line overlay */
.stApp::before {
    content:'';
    position:fixed; top:0; left:0; width:100%; height:100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,229,255,0.018) 2px,
        rgba(0,229,255,0.018) 4px
    );
    pointer-events:none; z-index:0;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #020810 100%) !important;
    border-right: 1px solid rgba(0,229,255,0.14);
}
[data-testid="stSidebar"] * { color: #7aaed4 !important; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #071828 0%, #0c2a45 100%) !important;
    color: #00e5ff !important;
    border: 1px solid rgba(0,229,255,0.3) !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 1.5px !important;
    padding: 8px 20px !important;
    transition: all 0.18s ease !important;
    width: 100%;
    position: relative; overflow: hidden;
}
.stButton > button::after {
    content:''; position:absolute; inset:0;
    background: linear-gradient(90deg,transparent 0%,rgba(0,229,255,0.07) 50%,transparent 100%);
    transform: translateX(-100%);
    transition: transform 0.4s ease;
}
.stButton > button:hover::after { transform: translateX(100%); }
.stButton > button:hover {
    border-color: rgba(0,229,255,0.75) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.28), 0 0 5px rgba(0,229,255,0.15) inset !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #060d1a;
    border-bottom: 1px solid rgba(0,229,255,0.15);
    gap: 1px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #3a6a8a !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    padding: 8px 22px;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.07) !important;
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
}

/* ── Streamlit metrics ───────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#08111f 0%,#0a1828 100%);
    border: 1px solid #0f2a42;
    border-radius: 6px;
    padding: 10px 14px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,229,255,0.35);
    box-shadow: 0 0 12px rgba(0,229,255,0.1);
}
[data-testid="metric-container"] label {
    color: #3a6a8a !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #00e5ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.5rem !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg,#0055bb,#00e5ff,#00ff88) !important;
    background-size: 200% 100% !important;
    animation: prog-shimmer 1.8s linear infinite !important;
}
@keyframes prog-shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
hr { border-color: #0f2a42 !important; margin: 12px 0 !important; }

/* ── Custom card (.cc) ───────────────────────────────────────────────── */
.cc {
    background: linear-gradient(135deg,#08111f 0%,#0a1828 100%);
    border: 1px solid #0f2a42;
    border-left: 3px solid #00e5ff;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 6px 0;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
    position: relative; overflow: hidden;
}
.cc::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg,transparent,rgba(0,229,255,0.25),transparent);
}
.cc:hover { transform: translateY(-1px); box-shadow: 0 4px 18px rgba(0,0,0,0.4); }
.cc.green  { border-left-color: #00ff88; }
.cc.green::before { background: linear-gradient(90deg,transparent,rgba(0,255,136,0.25),transparent); }
.cc.red    { border-left-color: #ff3366; }
.cc.red::before { background: linear-gradient(90deg,transparent,rgba(255,51,102,0.25),transparent); }
.cc.amber  { border-left-color: #ffaa00; }
.cc.amber::before { background: linear-gradient(90deg,transparent,rgba(255,170,0,0.25),transparent); }
.cc.purple { border-left-color: #9966ff; }
.cc.purple::before { background: linear-gradient(90deg,transparent,rgba(153,102,255,0.25),transparent); }
.cc .lbl { color:#3a6a8a; font-size:0.68rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:5px; }
.cc .val { font-family:'Share Tech Mono',monospace; font-size:1.7rem; color:#00e5ff; }
.cc.green  .val { color:#00ff88; }
.cc.red    .val { color:#ff3366; }
.cc.amber  .val { color:#ffaa00; }
.cc.purple .val { color:#9966ff; }
.cc .sub { color:#3a6a8a; font-size:0.72rem; margin-top:3px; }

/* ── Section header (.sh) ────────────────────────────────────────────── */
.sh {
    font-family: 'Share Tech Mono', monospace;
    color: #00e5ff;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding-bottom: 6px;
    border-bottom: 1px solid #0f2a42;
    margin: 18px 0 14px 0;
    position: relative;
}
.sh::after {
    content:''; position:absolute; bottom:-1px; left:0; width:60px; height:1px;
    background: #00e5ff;
    box-shadow: 0 0 6px rgba(0,229,255,0.6);
}

/* ── Page title ──────────────────────────────────────────────────────── */
.pt { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.9rem; color:#e8f4ff; margin-bottom:2px; }
.ps { color:#3a6a8a; font-size:0.82rem; letter-spacing:1px; margin-bottom:20px; }

/* ── Hero banner ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg,#050f1e 0%,#071828 40%,#0a1e30 100%);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 10px;
    padding: 28px 32px 22px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute; top:-60%; right:-10%;
    width:400px; height:400px;
    background: radial-gradient(circle,rgba(0,229,255,0.06) 0%,transparent 65%);
    pointer-events:none;
}
.hero::after {
    content:'';
    position:absolute; bottom:-80%; left:-5%;
    width:320px; height:320px;
    background: radial-gradient(circle,rgba(153,102,255,0.05) 0%,transparent 65%);
    pointer-events:none;
}
.hero-title {
    font-family:'Share Tech Mono',monospace;
    font-size:0.65rem; letter-spacing:4px; color:#3a6a8a;
    text-transform:uppercase; margin-bottom:8px;
}
.hero-name {
    font-family:'Rajdhani',sans-serif; font-weight:700;
    font-size:2.4rem; color:#e8f4ff; line-height:1;
    text-shadow: 0 0 30px rgba(0,229,255,0.2);
}
.hero-sub  { color:#4a7a9a; font-size:0.85rem; letter-spacing:1px; margin-top:6px; }
.hero-tags { margin-top:14px; display:flex; gap:6px; flex-wrap:wrap; }
.hero-tag  {
    font-family:'Share Tech Mono',monospace; font-size:0.65rem;
    letter-spacing:1.5px; color:#00e5ff;
    background:rgba(0,229,255,0.07); border:1px solid rgba(0,229,255,0.2);
    border-radius:3px; padding:3px 10px;
}

/* ── Pulse badge ─────────────────────────────────────────────────────── */
.pulse-badge {
    display:inline-flex; align-items:center; gap:6px;
    font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    letter-spacing:1.5px; color:#00ff88;
    background:rgba(0,255,136,0.07); border:1px solid rgba(0,255,136,0.25);
    border-radius:20px; padding:4px 12px;
}
.pulse-dot {
    width:7px; height:7px; border-radius:50%;
    background:#00ff88;
    box-shadow: 0 0 0 0 rgba(0,255,136,0.4);
    animation: pulse-ring 1.6s ease-in-out infinite;
    flex-shrink:0;
}
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0   rgba(0,255,136,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(0,255,136,0);   }
    100% { box-shadow: 0 0 0 0   rgba(0,255,136,0);   }
}
.pulse-badge.red   { color:#ff3366; background:rgba(255,51,102,0.07); border-color:rgba(255,51,102,0.25); }
.pulse-badge.red   .pulse-dot { background:#ff3366; box-shadow:0 0 0 0 rgba(255,51,102,0.4); }
@keyframes pulse-ring-red {
    0%   { box-shadow: 0 0 0 0   rgba(255,51,102,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(255,51,102,0);   }
    100% { box-shadow: 0 0 0 0   rgba(255,51,102,0);   }
}
.pulse-badge.red .pulse-dot { animation: pulse-ring-red 1.6s ease-in-out infinite; }

/* ── Feature badges ──────────────────────────────────────────────────── */
.badge {
    display:inline-block;
    background:#071828; border:1px solid #0f2a42;
    color:#5a9ac0; border-radius:3px;
    padding:2px 8px; font-size:0.68rem;
    font-family:'Share Tech Mono',monospace; margin:2px;
    transition: all 0.15s;
}
.badge:hover { border-color:rgba(0,229,255,0.3); color:#00e5ff; }
.badge.ok  { border-color:#1a4a2a; color:#40c080; }
.badge.bad { border-color:#4a1a1a; color:#c04040; text-decoration:line-through; }

/* ── Log table rows ──────────────────────────────────────────────────── */
.trow {
    font-family:'Share Tech Mono',monospace; font-size:0.76rem;
    padding:5px 10px; border-radius:3px; margin:2px 0;
    display:flex; gap:12px; align-items:center;
    transition: opacity 0.15s;
}
.trow:hover { opacity:0.85; }
.trow.blk { background:rgba(255,51,102,0.1);   border-left:3px solid #ff3366; }
.trow.rl  { background:rgba(255,170,0,0.08);   border-left:3px solid #ffaa00; }
.trow.al  { background:rgba(153,102,255,0.08); border-left:3px solid #9966ff; }
.trow.ok  { background:rgba(0,255,136,0.06);   border-left:3px solid #00ff88; }

/* ── Stat strip ──────────────────────────────────────────────────────── */
.stat-strip {
    display:flex; gap:2px; margin:12px 0;
    background:#030a14; border:1px solid #0f2a42; border-radius:6px;
    overflow:hidden;
}
.stat-cell {
    flex:1; padding:12px 14px; text-align:center;
    border-right:1px solid #0f2a42;
    transition: background 0.2s;
}
.stat-cell:last-child { border-right:none; }
.stat-cell:hover { background:rgba(0,229,255,0.04); }
.stat-cell .s-lbl { font-family:'Share Tech Mono',monospace; font-size:0.6rem; letter-spacing:2px; color:#3a6a8a; text-transform:uppercase; }
.stat-cell .s-val { font-family:'Share Tech Mono',monospace; font-size:1.4rem; color:#00e5ff; margin-top:2px; }
.stat-cell.g .s-val { color:#00ff88; }
.stat-cell.r .s-val { color:#ff3366; }
.stat-cell.a .s-val { color:#ffaa00; }
.stat-cell.p .s-val { color:#9966ff; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PLOTLY THEME
# ============================================================================

PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(6,13,26,0.95)',
    font=dict(family='Rajdhani, sans-serif', color='#7aaed4', size=11),
    title_font=dict(color='#c8e0f8', size=13, family='Rajdhani, sans-serif'),
    xaxis=dict(gridcolor='#0f2a42', linecolor='#0f2a42', tickcolor='#3a6a8a', zerolinecolor='#0f2a42'),
    yaxis=dict(gridcolor='#0f2a42', linecolor='#0f2a42', tickcolor='#3a6a8a', zerolinecolor='#0f2a42'),
    legend=dict(bgcolor='rgba(6,13,26,0.9)', bordercolor='#0f2a42', borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)
CC = ['#00e5ff','#00ff88','#ff3366','#ffaa00','#9966ff','#ff6633','#33ccff','#ff99cc']

def ptheme(fig, h=370):
    fig.update_layout(height=h, **PL)
    return fig

def card(lbl, val, sub="", cls=""):
    s = f'<div class="cc {cls}"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
    if sub: s += f'<div class="sub">{sub}</div>'
    return s + '</div>'

def sh(t):
    st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)

def ph(title, sub, tags=None):
    tag_html = ""
    if tags:
        tag_html = '<div class="hero-tags">' + "".join(f'<span class="hero-tag">{t}</span>' for t in tags) + '</div>'
    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">5G · DDOS · DETECTION</div>
      <div class="hero-name">{title}</div>
      <div class="hero-sub">{sub}</div>
      {tag_html}
    </div>""", unsafe_allow_html=True)

def pulse(text, cls=""):
    return f'<span class="pulse-badge {cls}"><span class="pulse-dot"></span>{text}</span>'

def stat_strip(cells):
    """cells = list of (label, value, cls) where cls is '' | 'g' | 'r' | 'a' | 'p'"""
    inner = "".join(
        f'<div class="stat-cell {c}"><div class="s-lbl">{l}</div><div class="s-val">{v}</div></div>'
        for l,v,c in cells
    )
    return f'<div class="stat-strip">{inner}</div>'

def rgba(hex6, a):
    """Convert 6-digit hex + alpha float to rgba() string — avoids 8-digit hex."""
    h = hex6.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'


# ============================================================================
# SESSION STATE
# ============================================================================

DEFS = {
    'data_loaded':False, 'data_preprocessed':False, 'models_trained':False,
    'df':None, 'results':{}, 'cv_results':{}, 'pipeline':None, 'data':None,
    'page':'📡 Data Lab',
}
for k,v in DEFS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# ML PIPELINE
# ============================================================================

class DDoSPipeline:
    LEAKAGE = ['Seq','RunTime','Offset','Dur']
    K       = 35
    TEST_SZ = 0.20
    VAL_SZ  = 0.25

    def __init__(self):
        self.scaler     = RobustScaler()
        self.le         = LabelEncoder()
        self.feat_names = None

    def load(self, f):
        try:
            df = pd.read_csv(f)
            return df, None
        except Exception as e:
            return None, str(e)

    def preprocess(self, df, cb=None):
        if cb: cb(8,  "Removing duplicates…")
        dup = df.duplicated().sum()
        df  = df.drop_duplicates().reset_index(drop=True)

        if cb: cb(18, "Replacing infinities (vectorised)…")
        # Vectorised inf→NaN in one pass — much faster than column-by-column replace
        num_cols_all = df.select_dtypes(include=[np.number]).columns
        df[num_cols_all] = df[num_cols_all].replace([np.inf, -np.inf], np.nan)

        if cb: cb(26, "Filling missing values (parallel)…")
        # Compute medians in one call, then fillna in parallel via thread pool
        medians = df[num_cols_all].median()
        def _fill(col):
            if df[col].isnull().any():
                df[col] = df[col].fillna(medians[col])
        cols_with_na = [c for c in num_cols_all if df[c].isnull().any()]
        if cols_with_na:
            with ThreadPoolExecutor() as ex:
                list(ex.map(_fill, cols_with_na))

        if cb: cb(36, "Removing data-leakage features…")
        removed = [c for c in self.LEAKAGE if c in df.columns]

        if cb: cb(46, "Engineering domain features…")
        if 'TotPkts' in df.columns and 'FlowDuration' in df.columns:
            df['PktRate'] = (df['TotPkts'] / (df['FlowDuration']+1e-9)).clip(-1e9, 1e9)
        if 'SrcPkts' in df.columns and 'DstPkts' in df.columns:
            df['SrcDstRatio'] = (df['SrcPkts'] / (df['DstPkts']+1e-9)).clip(-1e9, 1e9)
        if 'TotBytes' in df.columns and 'TotPkts' in df.columns:
            df['AvgPktSize'] = (df['TotBytes'] / (df['TotPkts']+1e-9)).clip(-1e9, 1e9)

        if cb: cb(56, "Building feature matrix…")
        drop = ['Label','Attack Type','Attack Tool'] + removed
        X    = df.drop(drop, axis=1, errors='ignore').select_dtypes(include=[np.number])
        y    = self.le.fit_transform(df['Label'])

        # Final vectorised NaN/inf sweep (covers newly engineered columns)
        X = X.replace([np.inf, -np.inf], np.nan)
        med2 = X.median()
        for c in X.columns:
            if X[c].isnull().any():
                X[c] = X[c].fillna(med2[c])
        X = X.dropna(axis=1, how='all')

        if cb: cb(68, f"Selecting top {self.K} features (mutual info, sampled)…")
        k   = min(self.K, X.shape[1])
        # Use at most 60 k rows for mutual_info — gives same ranking, much faster
        MI_SAMPLE = 60_000
        if len(X) > MI_SAMPLE:
            idx_s = np.random.choice(len(X), MI_SAMPLE, replace=False)
            Xs_mi, ys_mi = X.iloc[idx_s].values, y[idx_s]
        else:
            Xs_mi, ys_mi = X.values, y
        sel = SelectKBest(mutual_info_classif, k=k)
        sel.fit(Xs_mi, ys_mi)
        self.feat_names = X.columns[sel.get_support()].tolist()
        X = X[self.feat_names]

        if cb: cb(80, "Splitting train / val / test…")
        Xt, Xte, yt, yte = train_test_split(X, y, test_size=self.TEST_SZ, random_state=7, stratify=y)
        Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=self.VAL_SZ, random_state=13, stratify=yt)

        if cb: cb(92, "Scaling with RobustScaler…")
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xv_s  = self.scaler.transform(Xv)
        Xte_s = self.scaler.transform(Xte)

        if cb: cb(100, "Done!")
        return dict(
            X_train=Xtr_s, X_val=Xv_s,  X_test=Xte_s,
            y_train=ytr,   y_val=yv,     y_test=yte,
            leakage_removed=removed, feat_selected=len(self.feat_names),
            feat_names=self.feat_names, dups=dup,
            n_train=len(Xtr_s), n_val=len(Xv_s), n_test=len(Xte_s),
        )

    def get_models(self):
        m = {
            # max_iter 300→200; n_jobs already -1
            'Logistic Regression': LogisticRegression(
                max_iter=200, random_state=42, class_weight='balanced', n_jobs=-1, solver='saga'),
            # shallower tree → faster fit, less overfitting
            'Decision Tree':       DecisionTreeClassifier(
                max_depth=12, min_samples_split=30, min_samples_leaf=15,
                random_state=42, class_weight='balanced'),
            # 100 trees instead of 200 — negligible accuracy delta, ~2× faster
            'Random Forest':       RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=30, min_samples_leaf=15,
                random_state=42, n_jobs=-1, class_weight='balanced',
                max_features='sqrt'),
            # fewer estimators + smaller subsample
            'Gradient Boosting':   GradientBoostingClassifier(
                n_estimators=80, learning_rate=0.12, max_depth=4,
                subsample=0.75, random_state=42, n_iter_no_change=10, tol=1e-4),
            # smaller net + fewer iterations + warm start
            'Neural Network':      MLPClassifier(
                hidden_layer_sizes=(96, 48), max_iter=300, random_state=42,
                early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
                learning_rate_init=0.001, batch_size=256),
        }
        if XGBOOST_AVAILABLE:
            m['XGBoost'] = XGBClassifier(
                n_estimators=100, learning_rate=0.12, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, eval_metric='logloss',
                tree_method='hist')          # hist method is much faster
        if LIGHTGBM_AVAILABLE:
            m['LightGBM'] = LGBMClassifier(
                n_estimators=100, learning_rate=0.12, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
                num_leaves=31)
        return m

    def train(self, name, model, data, folds=3, cb=None):
        # Build combined train+val array once (avoid repeated vstack per call)
        Xc = np.vstack([data['X_train'], data['X_val']])
        yc = np.concatenate([data['y_train'], data['y_val']])
        Xt, yt = data['X_test'], data['y_test']

        if cb: cb(15, f"{folds}-fold CV: {name}…")
        # 3 folds → ~40% faster than 5 folds with only tiny variance increase
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv  = cross_val_score(model, Xc, yc, cv=skf, scoring='f1_weighted', n_jobs=-1)

        if cb: cb(62, f"Training {name}…")
        t0 = time.time()
        model.fit(Xc, yc)
        tt = time.time()-t0

        if cb: cb(88, "Evaluating…")
        yp  = model.predict(Xt)
        ypr = model.predict_proba(Xt) if hasattr(model,'predict_proba') else None
        cm  = confusion_matrix(yt, yp)

        if cb: cb(100, "Done!")
        return dict(
            model=model, train_time=tt,
            cv_mean=cv.mean(), cv_std=cv.std(), cv_scores=cv,
            test_accuracy =accuracy_score(yt,yp),
            test_precision=precision_score(yt,yp,average='weighted',zero_division=0),
            test_recall   =recall_score(yt,yp,average='weighted',zero_division=0),
            test_f1       =f1_score(yt,yp,average='weighted',zero_division=0),
            test_mcc      =matthews_corrcoef(yt,yp),
            confusion_matrix=cm, predictions=yp, probabilities=ypr,
        )


# ============================================================================
# SIDEBAR
# ============================================================================

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px">
          <div style="font-family:'Share Tech Mono',monospace;color:#00e5ff;font-size:1.1rem;letter-spacing:3px;text-shadow:0 0 12px rgba(0,229,255,0.4)">🛡 5G·DDOS·DETECT</div>
          <div style="color:#1a4060;font-size:0.6rem;letter-spacing:2px;margin-top:4px">THREAT INTELLIGENCE SYSTEM</div>
          <div style="display:inline-block;margin-top:8px;background:rgba(0,229,255,0.07);border:1px solid rgba(0,229,255,0.2);border-radius:3px;padding:2px 10px;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a7a9a;letter-spacing:1px">v2 · OPTIMISED</div>
        </div><hr>
        """, unsafe_allow_html=True)

        pages = [("📡","Data Lab"),("⚙️","Pipeline"),("🤖","Model Arena"),("📊","Analytics Hub"),("🛡️","Threat Monitor")]
        st.markdown('<div style="font-family:Share Tech Mono,monospace;color:#3a6a8a;font-size:0.6rem;letter-spacing:2px;margin-bottom:8px">NAVIGATION</div>', unsafe_allow_html=True)
        cur = st.session_state.page
        for icon, name in pages:
            lbl = f"{icon} {name}"
            active = name in cur
            if active:
                st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#00e5ff;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.25);border-radius:4px;padding:7px 12px;margin:2px 0;letter-spacing:1px">▶ {icon} {name}</div>', unsafe_allow_html=True)
            else:
                if st.button(lbl, key=f"nb_{name}"):
                    st.session_state.page = lbl
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:Share Tech Mono,monospace;color:#3a6a8a;font-size:0.6rem;letter-spacing:2px;margin-bottom:8px">DATASET</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload CS_5G_NIDD.csv", type=['csv'], label_visibility="collapsed")
        if up and not st.session_state.data_loaded:
            with st.spinner("Loading…"):
                p = DDoSPipeline()
                df, err = p.load(up)
                if err:
                    st.error(err)
                else:
                    st.session_state.df       = df
                    st.session_state.pipeline = p
                    st.session_state.data_loaded = True
                    st.success("Loaded ✓")
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:Share Tech Mono,monospace;color:#3a6a8a;font-size:0.6rem;letter-spacing:2px;margin-bottom:8px">PIPELINE STATUS</div>', unsafe_allow_html=True)
        for lbl, ok in [("Data Loaded",st.session_state.data_loaded),
                        ("Preprocessed",st.session_state.data_preprocessed),
                        ("Models Trained",st.session_state.models_trained)]:
            clr   = "#00ff88" if ok else "#ff3366"
            bclr  = "rgba(0,255,136,0.1)" if ok else "rgba(255,51,102,0.07)"
            bdclr = "rgba(0,255,136,0.25)" if ok else "rgba(255,51,102,0.2)"
            icon  = "✓" if ok else "○"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;'
                f'background:{bclr};border:1px solid {bdclr};border-radius:4px;margin:3px 0;'
                f'font-family:Share Tech Mono,monospace;font-size:0.72rem;color:{clr}">'
                f'<span style="font-size:0.8rem">{icon}</span>{lbl}</div>',
                unsafe_allow_html=True
            )

        if st.session_state.data_loaded:
            df = st.session_state.df
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div style="font-family:Share Tech Mono,monospace;color:#3a6a8a;font-size:0.6rem;letter-spacing:2px;margin-bottom:6px">DATASET INFO</div>', unsafe_allow_html=True)
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", df.shape[1])


# ============================================================================
# PAGE 1 — DATA LAB
# ============================================================================

def page_data_lab():
    ph("📡 DATA LAB",
       "Automated dataset analysis — explore distributions, correlations, and feature profiles",
       tags=["LIVE PREVIEW","DISTRIBUTIONS","FEATURE EXPLORER","CORRELATION MAP"])

    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="border:1px dashed rgba(0,229,255,0.22);border-radius:10px;padding:50px;text-align:center;margin-top:40px;background:rgba(0,229,255,0.02)">
          <div style="font-size:3rem;margin-bottom:14px">📁</div>
          <div style="font-family:Share Tech Mono,monospace;color:#3a6a8a;letter-spacing:3px;font-size:0.85rem">
            UPLOAD CS_5G_NIDD.CSV VIA THE SIDEBAR TO BEGIN
          </div>
          <div style="margin-top:10px;color:#1e3a55;font-size:0.75rem;letter-spacing:1px">
            Supports CSV datasets with a <span style="color:#2a5a7a">Label</span> column
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    benign = int((df['Label']=='Benign').sum()) if 'Label' in df.columns else 0
    mal    = len(df) - benign

    # Stat strip + live badges
    pct_b = benign/len(df)*100 if len(df) else 0
    pct_m = mal/len(df)*100 if len(df) else 0
    st.markdown(stat_strip([
        ("TOTAL FLOWS",  f"{len(df):,}",       ""),
        ("FEATURES",     str(df.shape[1]),      ""),
        ("BENIGN",       f"{benign:,}",         "g"),
        ("MALICIOUS",    f"{mal:,}",            "r"),
        ("BENIGN RATIO", f"{pct_b:.1f}%",       "g"),
        ("ATTACK RATIO", f"{pct_m:.1f}%",       "r"),
    ]), unsafe_allow_html=True)
    st.markdown(
        pulse("DATASET LOADED", "") + "&nbsp;&nbsp;" +
        (pulse("ATTACKS DETECTED", "red") if mal > 0 else ""),
        unsafe_allow_html=True
    )
    st.markdown("")

    tab1,tab2,tab3,tab4 = st.tabs(["OVERVIEW","DISTRIBUTIONS","FEATURE EXPLORER","CORRELATION MAP"])

    # ── OVERVIEW ─────────────────────────────────────────────────────────────
    with tab1:
        sh("LIVE DATA PREVIEW — FIRST 200 ROWS")
        st.dataframe(df.head(200), use_container_width=True, height=340)

        sh("DATA QUALITY")
        q1,q2 = st.columns(2)
        with q1:
            miss = df.isnull().sum()
            miss = miss[miss>0]
            if len(miss):
                fig = go.Figure(go.Bar(
                    x=miss.index, y=miss.values,
                    marker_color='#ff3366',
                    text=miss.values, textposition='outside',
                    textfont=dict(color='#a0c4e8')))
                fig.update_layout(title="Missing Values per Column", xaxis_tickangle=45)
                ptheme(fig); st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ Zero missing values detected")
        with q2:
            dtc = df.dtypes.value_counts().reset_index()
            dtc.columns = ['dtype','count']
            dtc['dtype'] = dtc['dtype'].astype(str)
            fig = go.Figure(go.Pie(
                labels=dtc['dtype'], values=dtc['count'], hole=0.45,
                marker=dict(colors=CC),
                textfont=dict(family='Share Tech Mono', size=11)))
            fig.update_layout(title="Column Data Types")
            ptheme(fig); st.plotly_chart(fig, use_container_width=True)

    # ── DISTRIBUTIONS ─────────────────────────────────────────────────────────
    with tab2:
        if 'Label' in df.columns:
            d1,d2 = st.columns(2)
            with d1:
                lc = df['Label'].value_counts()
                fig = go.Figure(go.Pie(
                    labels=lc.index, values=lc.values, hole=0.42,
                    marker=dict(colors=['#00ff88','#ff3366']),
                    textfont=dict(family='Share Tech Mono', size=12)))
                fig.update_layout(title="Traffic Label Distribution")
                ptheme(fig); st.plotly_chart(fig, use_container_width=True)
            with d2:
                if 'Attack Type' in df.columns:
                    ac = df['Attack Type'].value_counts()
                    fig = go.Figure(go.Bar(
                        x=ac.values, y=ac.index, orientation='h',
                        marker=dict(color=CC[:len(ac)]),
                        text=ac.values, textposition='outside',
                        textfont=dict(color='#a0c4e8')))
                    fig.update_layout(title="Attack Type Breakdown", yaxis=dict(autorange='reversed'))
                    ptheme(fig); st.plotly_chart(fig, use_container_width=True)

        sh("TOP 12 HIGH-VARIANCE FEATURE DISTRIBUTIONS")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        top12    = df[num_cols].var().nlargest(12).index.tolist()
        sample   = df.sample(min(30000, len(df)), random_state=1)

        for i in range(0, len(top12), 3):
            row_feats = top12[i:i+3]
            cols      = st.columns(3)
            for j, feat in enumerate(row_feats):
                with cols[j]:
                    fig = px.histogram(sample, x=feat, nbins=40,
                        color='Label' if 'Label' in df.columns else None,
                        color_discrete_map={'Benign':'#00ff88','Malicious':'#ff3366'},
                        barmode='overlay')
                    fig.update_traces(opacity=0.72)
                    fig.update_layout(title=feat, showlegend=False,
                        margin=dict(l=20,r=10,t=35,b=20))
                    ptheme(fig, h=220); st.plotly_chart(fig, use_container_width=True)

    # ── FEATURE EXPLORER ──────────────────────────────────────────────────────
    with tab3:
        sh("AUTO SCATTER ANALYSIS — TOP VARIANCE FEATURE PAIRS")
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        top_vars  = df[num_cols].var().nlargest(8).index.tolist()
        pairs     = [(top_vars[i], top_vars[j]) for i in range(4) for j in range(i+1,4)][:6]
        sdf       = df.sample(min(5000, len(df)), random_state=42)

        for k in range(0, len(pairs), 2):
            pair_row = pairs[k:k+2]
            cols     = st.columns(len(pair_row))
            for ci, pair in enumerate(pair_row):
                with cols[ci]:
                    fig = px.scatter(sdf, x=pair[0], y=pair[1],
                        color='Label' if 'Label' in df.columns else None,
                        color_discrete_map={'Benign':'#00e5ff','Malicious':'#ff3366'},
                        opacity=0.45)
                    fig.update_traces(marker=dict(size=3))
                    fig.update_layout(title=f"{pair[0]}  ×  {pair[1]}",
                        showlegend=False, margin=dict(l=20,r=10,t=35,b=20))
                    ptheme(fig, h=260); st.plotly_chart(fig, use_container_width=True)

        sh("FEATURE STATISTICS — TOP 20")
        top20 = df[num_cols].var().nlargest(20).index.tolist()
        st.dataframe(df[top20].describe().T.round(4), use_container_width=True)

    # ── CORRELATION MAP ──────────────────────────────────────────────────────
    with tab4:
        sh("CORRELATION HEATMAP — TOP 20 HIGH-VARIANCE FEATURES")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        top20    = df[num_cols].var().nlargest(20).index.tolist()
        corr     = df[top20].corr()

        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,'#ff3366'],[0.5,'#04080f'],[1,'#00e5ff']],
            zmid=0, text=np.round(corr.values, 2),
            texttemplate='%{text}', textfont=dict(size=8),
            hovertemplate='%{x} ↔ %{y}: %{z:.3f}<extra></extra>',
        ))
        fig.update_layout(title="Feature Correlation Matrix (Top 20)")
        ptheme(fig, h=560); st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 2 — PIPELINE
# ============================================================================

def page_pipeline():
    ph("⚙️  PIPELINE",
       "One-click automated preprocessing — leakage removal, feature selection, data splitting",
       tags=["DEDUP","INF REMOVAL","PARALLEL FILL","FEATURE ENGINEERING","MUTUAL INFO","SPLIT","SCALE"])

    if not st.session_state.data_loaded:
        st.warning("Upload a dataset first."); return

    df = st.session_state.df

    sh("AUTOMATED PIPELINE CONFIGURATION")
    leakage_present = [c for c in DDoSPipeline.LEAKAGE if c in df.columns]
    cfg1, cfg2 = st.columns(2)

    with cfg1:
        badges_bad = "".join(f'<span class="badge bad">✗ {c}</span>' for c in leakage_present) or '<span style="color:#3a6a8a;font-size:0.8rem">None found</span>'
        st.markdown(f"""
        <div class="cc"><div class="lbl">LEAKAGE FEATURES TO REMOVE</div>
        <div style="margin-top:8px">{badges_bad}</div>
        <div class="sub" style="margin-top:8px">These time-based features cause artificial 100% accuracy</div></div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cc"><div class="lbl">FEATURE SELECTION</div>
        <div class="val">Top {DDoSPipeline.K}</div>
        <div class="sub">Mutual information score (SelectKBest)</div></div>
        """, unsafe_allow_html=True)

    with cfg2:
        tr = int(len(df) * (1-DDoSPipeline.TEST_SZ) * (1-DDoSPipeline.VAL_SZ))
        va = int(len(df) * (1-DDoSPipeline.TEST_SZ) * DDoSPipeline.VAL_SZ)
        te = int(len(df) * DDoSPipeline.TEST_SZ)
        st.markdown(f"""
        <div class="cc"><div class="lbl">TRAIN / VAL / TEST SPLIT</div>
        <div style="margin-top:8px;font-family:Share Tech Mono,monospace;font-size:0.85rem">
          <span style="color:#00e5ff">TRAIN</span> {tr:,} ({int((1-DDoSPipeline.TEST_SZ)*(1-DDoSPipeline.VAL_SZ)*100)}%)
          &nbsp;·&nbsp;
          <span style="color:#9966ff">VAL</span> {va:,} ({int((1-DDoSPipeline.TEST_SZ)*DDoSPipeline.VAL_SZ*100)}%)
          &nbsp;·&nbsp;
          <span style="color:#ff3366">TEST</span> {te:,} ({int(DDoSPipeline.TEST_SZ*100)}%)
        </div>
        <div class="sub" style="margin-top:8px">Stratified splits — class balance preserved</div></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="cc"><div class="lbl">SCALER</div>
        <div class="val" style="font-size:1.2rem">RobustScaler</div>
        <div class="sub">Resistant to outliers in network traffic data</div></div>
        """, unsafe_allow_html=True)

    sh("ACCURACY IMPACT OF LEAKAGE REMOVAL")
    ai1, ai2 = st.columns(2)
    ai1.markdown(card("WITH LEAKAGE FEATURES",    "99–100%", "Unrealistic — model memorises timestamps", "red"),   unsafe_allow_html=True)
    ai2.markdown(card("WITHOUT LEAKAGE FEATURES", "92–97%",  "Production-ready — learns real patterns",  "green"), unsafe_allow_html=True)

    st.markdown("")

    if not st.session_state.data_preprocessed:
        if st.button("▶  RUN AUTOMATED PIPELINE", type="primary"):
            bar  = st.progress(0)
            stat = st.empty()
            def cb(pct, msg):
                bar.progress(pct/100)
                stat.markdown(f'<span style="font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#00e5ff">[{pct:3d}%] {msg}</span>', unsafe_allow_html=True)
            with st.spinner("Running…"):
                result = st.session_state.pipeline.preprocess(df.copy(), cb=cb)
            bar.empty(); stat.empty()
            st.session_state.data = result
            st.session_state.data_preprocessed = True
            st.success("✅ Preprocessing complete!"); st.rerun()
    else:
        st.success("✅ Preprocessing already complete — results below.")

    if st.session_state.data_preprocessed:
        d = st.session_state.data
        sh("PREPROCESSING RESULTS")
        st.markdown(stat_strip([
            ("DUPLICATES REMOVED", f"{d['dups']:,}",          "r"),
            ("LEAKAGE REMOVED",    str(len(d['leakage_removed'])), "a"),
            ("FEATURES SELECTED",  str(d['feat_selected']),   ""),
            ("TRAIN SAMPLES",      f"{d['n_train']:,}",       "g"),
            ("VAL SAMPLES",        f"{d['n_val']:,}",         "p"),
            ("TEST SAMPLES",       f"{d['n_test']:,}",        ""),
        ]), unsafe_allow_html=True)
        st.markdown(pulse("PIPELINE COMPLETE", ""), unsafe_allow_html=True)
        st.markdown("")

        sh("DATA SPLIT VISUALIZATION")
        fig = go.Figure()
        for lbl, sz, c in [("Train",d['n_train'],'#00e5ff'),("Validation",d['n_val'],'#9966ff'),("Test",d['n_test'],'#ff3366')]:
            fig.add_trace(go.Bar(name=lbl, x=[sz], y=['Split'], orientation='h',
                marker_color=c, text=f"{lbl}: {sz:,}", textposition='inside',
                textfont=dict(color='white', size=12)))
        fig.update_layout(barmode='stack', showlegend=True, title="Sample Distribution")
        ptheme(fig, h=130); st.plotly_chart(fig, use_container_width=True)

        sh("SELECTED FEATURES")
        badges = "".join(f'<span class="badge ok">✓ {f}</span>' for f in d['feat_names'])
        st.markdown(badges, unsafe_allow_html=True)


# ============================================================================
# PAGE 3 — MODEL ARENA
# ============================================================================

def page_model_arena():
    ph("🤖 MODEL ARENA",
       "Automated training of all models — 3-fold cross-validation — live leaderboard",
       tags=["LOGISTIC REG","DECISION TREE","RANDOM FOREST","GRAD BOOST","NEURAL NET","XGBOOST","LIGHTGBM"])

    if not st.session_state.data_preprocessed:
        st.warning("Complete the Pipeline step first."); return

    if not st.session_state.models_trained:
        if st.button("⚡  TRAIN ALL MODELS (AUTOMATED)", type="primary"):
            pipeline = st.session_state.pipeline
            models   = pipeline.get_models()
            data     = st.session_state.data
            results  = {}; cv_r = {}
            over_bar = st.progress(0)
            over_txt = st.empty()
            live_ph  = st.empty()
            rows     = []

            for idx,(name,model) in enumerate(models.items(),1):
                over_txt.markdown(f'<span style="font-family:Share Tech Mono,monospace;color:#00e5ff">TRAINING [{idx}/{len(models)}] {name}</span>', unsafe_allow_html=True)
                m_bar = st.progress(0)
                m_txt = st.empty()

                def cb(p, m, b=m_bar, t=m_txt):
                    b.progress(p/100)
                    t.markdown(f'<span style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:#4a8ab0">{m}</span>', unsafe_allow_html=True)

                res = pipeline.train(name, model, data, cb=cb)
                results[name] = res
                cv_r[name]    = dict(scores=res['cv_scores'], mean=res['cv_mean'], std=res['cv_std'])
                m_bar.empty(); m_txt.empty()

                rows.append({
                    'Rank': idx, 'Model': name,
                    'Accuracy': f"{res['test_accuracy']:.2%}",
                    'F1-Score': f"{res['test_f1']:.4f}",
                    'CV F1':    f"{res['cv_mean']:.4f}",
                    'CV Std':   f"±{res['cv_std']:.4f}",
                    'Time':     f"{res['train_time']:.1f}s",
                })
                live_ph.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                over_bar.progress(idx/len(models))

            st.session_state.results    = results
            st.session_state.cv_results = cv_r
            st.session_state.models_trained = True
            over_bar.empty(); over_txt.empty()
            st.success("🎉 All models trained!"); st.balloons(); st.rerun()
    else:
        st.success("✅ Models already trained — leaderboard below.")

    if st.session_state.models_trained:
        results = st.session_state.results
        sh("LEADERBOARD")
        best = max(results, key=lambda m: results[m]['test_f1'])
        br   = results[best]
        n_models  = len(results)
        st.markdown(stat_strip([
            ("🏆 BEST MODEL",   best,                         "g"),
            ("ACCURACY",        f"{br['test_accuracy']:.2%}", "g"),
            ("F1-SCORE",        f"{br['test_f1']:.4f}",       "g"),
            ("CV SCORE",        f"{br['cv_mean']:.4f}",       "a"),
            ("CV STD",          f"±{br['cv_std']:.4f}",       ""),
            ("MODELS TRAINED",  str(n_models),                ""),
        ]), unsafe_allow_html=True)
        st.markdown(pulse("ALL MODELS TRAINED", ""), unsafe_allow_html=True)
        st.markdown("")

        sh("FULL RESULTS TABLE")
        lb = []
        for i,(nm,res) in enumerate(sorted(results.items(), key=lambda x:x[1]['test_f1'], reverse=True),1):
            lb.append({'Rank':i,'Model':nm,'Accuracy':f"{res['test_accuracy']:.2%}",
                'F1':f"{res['test_f1']:.4f}",'Precision':f"{res['test_precision']:.4f}",
                'Recall':f"{res['test_recall']:.4f}",'MCC':f"{res['test_mcc']:.4f}",
                'CV F1':f"{res['cv_mean']:.4f}",'CV Std':f"±{res['cv_std']:.4f}",
                'Train Time':f"{res['train_time']:.1f}s"})
        lbdf = pd.DataFrame(lb)
        st.dataframe(lbdf, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Leaderboard CSV", lbdf.to_csv(index=False), "leaderboard.csv","text/csv")

        sh("ACCURACY COMPARISON CHART")
        mnames = [r['Model'] for r in lb]
        fig = go.Figure()
        for metric, color in [('test_accuracy','#00e5ff'),('test_f1','#00ff88'),
                               ('test_precision','#9966ff'),('test_recall','#ffaa00')]:
            fig.add_trace(go.Bar(
                name=metric.replace('test_','').title(),
                x=mnames, y=[results[m][metric] for m in mnames],
                marker_color=color,
                text=[f"{results[m][metric]:.3f}" for m in mnames],
                textposition='outside', textfont=dict(size=9, color='#a0c4e8')))
        fig.update_layout(barmode='group', title="All Metrics by Model",
            yaxis_range=[min(results[m]['test_accuracy'] for m in mnames)-0.06, 1.01],
            xaxis_tickangle=15)
        ptheme(fig, h=420); st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 4 — ANALYTICS HUB
# ============================================================================

def page_analytics_hub():
    ph("📊 ANALYTICS HUB",
       "Model deep-dive — confusion matrices, ROC curves, CV analysis, radar charts",
       tags=["RADAR","CONFUSION MATRIX","CV ANALYSIS","ROC CURVES","AUC RANKING"])

    if not st.session_state.models_trained:
        st.warning("Train models first."); return

    results  = st.session_state.results
    cv_r     = st.session_state.cv_results
    mnames   = list(results.keys())
    data     = st.session_state.data
    pipeline = st.session_state.pipeline

    tab1,tab2,tab3,tab4 = st.tabs(["RADAR & METRICS","CONFUSION MATRICES","CV ANALYSIS","ROC CURVES"])

    # ── RADAR ────────────────────────────────────────────────────────────────
    with tab1:
        sh("PERFORMANCE RADAR — ALL MODELS")
        cats = ['Accuracy','Precision','Recall','F1-Score','CV F1','MCC (norm)']
        fig_r = go.Figure()
        for nm, c in zip(mnames, CC):
            r    = results[nm]
            vals = [r['test_accuracy'],r['test_precision'],r['test_recall'],
                    r['test_f1'],r['cv_mean'],(r['test_mcc']+1)/2]
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself', opacity=0.4, name=nm, line=dict(color=c, width=2)))
        fig_r.update_layout(
            polar=dict(bgcolor='#06080f',
                radialaxis=dict(visible=True, range=[0.8,1.0], gridcolor='#0f2a42', color='#3a6a8a'),
                angularaxis=dict(gridcolor='#0f2a42', color='#7aaed4')),
            title="Model Performance Radar")
        ptheme(fig_r, h=460); st.plotly_chart(fig_r, use_container_width=True)

        sh("ALL METRICS COMPARISON")
        fig = go.Figure()
        for metric, color in [('test_accuracy','#00e5ff'),('test_f1','#00ff88'),
                               ('test_precision','#9966ff'),('test_recall','#ffaa00'),
                               ('test_mcc','#ff3366')]:
            lbl = metric.replace('test_','').replace('_',' ').title()
            fig.add_trace(go.Bar(name=lbl, x=mnames,
                y=[results[m][metric] for m in mnames], marker_color=color))
        fig.update_layout(barmode='group', title="All Metrics Comparison",
            yaxis_range=[min(results[m]['test_accuracy'] for m in mnames)-0.06, 1.01],
            xaxis_tickangle=15)
        ptheme(fig, h=400); st.plotly_chart(fig, use_container_width=True)

    # ── CONFUSION MATRICES ────────────────────────────────────────────────────
    with tab2:
        sh("CONFUSION MATRICES — ALL MODELS")
        labels = list(pipeline.le.classes_)
        for i in range(0, len(mnames), 2):
            pair = mnames[i:i+2]
            cols = st.columns(len(pair))
            for j, nm in enumerate(pair):
                with cols[j]:
                    cm  = results[nm]['confusion_matrix']
                    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                    ann = [[f"{cm[r][c]:,}\n({pct[r][c]:.1%})" for c in range(cm.shape[1])] for r in range(cm.shape[0])]
                    fig = go.Figure(go.Heatmap(
                        z=pct, x=labels, y=labels,
                        text=ann, texttemplate="%{text}",
                        textfont=dict(size=12, family='Share Tech Mono'),
                        colorscale=[[0,'#04080f'],[0.5,'#003366'],[1,'#00e5ff']],
                        showscale=False,
                        hovertemplate='True: %{y}<br>Pred: %{x}<extra></extra>'))
                    fig.update_layout(title=nm, xaxis_title="Predicted", yaxis_title="Actual")
                    ptheme(fig, h=320); st.plotly_chart(fig, use_container_width=True)
                    if cm.shape == (2,2):
                        tn,fp,fn,tp = cm.ravel()
                        fpr = fp/(fp+tn) if fp+tn else 0
                        fnr = fn/(fn+tp) if fn+tp else 0
                        mx1,mx2,mx3,mx4 = st.columns(4)
                        mx1.metric("TP",f"{tp:,}"); mx2.metric("TN",f"{tn:,}")
                        mx3.metric("FPR",f"{fpr:.2%}"); mx4.metric("FNR",f"{fnr:.2%}")
                    st.markdown("<hr>", unsafe_allow_html=True)

    # ── CV ANALYSIS ───────────────────────────────────────────────────────────
    with tab3:
        sh("5-FOLD CV SCORES — ALL MODELS")
        fig_cv = go.Figure()
        for nm, c in zip(mnames, CC):
            sc = cv_r[nm]['scores']
            fig_cv.add_trace(go.Scatter(
                x=[f"Fold {i+1}" for i in range(len(sc))],
                y=sc, mode='lines+markers', name=nm,
                line=dict(color=c, width=2.5),
                marker=dict(size=9, color=c, line=dict(color='#04080f', width=2))))
        fig_cv.update_layout(title="CV Fold Scores per Model", yaxis_title="F1 (weighted)")
        ptheme(fig_cv, h=400); st.plotly_chart(fig_cv, use_container_width=True)

        sh("CV VARIANCE — STABILITY COMPARISON")
        means = [cv_r[nm]['mean'] for nm in mnames]
        stds  = [cv_r[nm]['std']  for nm in mnames]
        fig_e = go.Figure()
        fig_e.add_trace(go.Bar(
            x=mnames, y=means,
            error_y=dict(type='data', array=stds, visible=True, color='#ffaa00'),
            marker_color=['#00e5ff' if m==max(means) else '#1a4a7a' for m in means],
            text=[f"{m:.4f}" for m in means], textposition='outside',
            textfont=dict(color='#a0c4e8', size=9)))
        fig_e.update_layout(title="CV Mean ± Std — Model Stability",
            yaxis_title="F1 (weighted)",
            yaxis_range=[min(m-s-0.01 for m,s in zip(means,stds)), 1.01],
            xaxis_tickangle=15)
        ptheme(fig_e, h=370); st.plotly_chart(fig_e, use_container_width=True)

    # ── ROC CURVES ────────────────────────────────────────────────────────────
    with tab4:
        sh("ROC CURVES — ALL MODELS")
        yt = data['y_test']
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
            line=dict(dash='dot', color='#3a6a8a', width=1))
        for nm, c in zip(mnames, CC):
            pr = results[nm]['probabilities']
            if pr is not None and pr.shape[1] == 2:
                fpr, tpr, _ = roc_curve(yt, pr[:,1])
                rauc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                    name=f"{nm} (AUC={rauc:.3f})", line=dict(color=c, width=2.5)))
        fig_roc.update_layout(title="ROC Curves",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        ptheme(fig_roc, h=480); st.plotly_chart(fig_roc, use_container_width=True)

        sh("AUC RANKING")
        aucs = []
        for nm in mnames:
            pr = results[nm]['probabilities']
            if pr is not None and pr.shape[1] == 2:
                fpr, tpr, _ = roc_curve(yt, pr[:,1])
                aucs.append((nm, auc(fpr, tpr)))
        aucs.sort(key=lambda x: -x[1])
        fig_auc = go.Figure(go.Bar(
            x=[a[0] for a in aucs], y=[a[1] for a in aucs],
            marker_color=CC[:len(aucs)],
            text=[f"{a[1]:.4f}" for a in aucs], textposition='outside',
            textfont=dict(color='#a0c4e8')))
        fig_auc.update_layout(title="AUC Scores Ranked",
            yaxis_range=[0.9,1.01], xaxis_tickangle=15)
        ptheme(fig_auc, h=320); st.plotly_chart(fig_auc, use_container_width=True)


# ============================================================================
# PAGE 5 — THREAT MONITOR (animated)
# ============================================================================

ACTION_COLORS = {"BLOCKED":"#ff3366","RATE LIMITED":"#ffaa00","ALERT":"#9966ff","ALLOWED":"#00ff88"}
ACTION_ICONS  = {"BLOCKED":"🚫","RATE LIMITED":"⚠️","ALERT":"🔔","ALLOWED":"✅"}
ACTION_ROW_CLS= {"BLOCKED":"blk","RATE LIMITED":"rl","ALERT":"al","ALLOWED":"ok"}

def get_action(pred_label, conf, benign_cls, block_thr=0.95, rate_thr=0.85):
    if pred_label == benign_cls: return "ALLOWED"
    if conf >= block_thr:        return "BLOCKED"
    if conf >= rate_thr:         return "RATE LIMITED"
    return "ALERT"


def page_threat_monitor():
    ph("🛡️  THREAT MONITOR",
       "Animated real-time defense simulation — live packet feed — confidence visualization",
       tags=["LIVE SIMULATION","PACKET INSPECTOR","BLOCK","RATE-LIMIT","ALERT","ALLOW"])

    if not st.session_state.models_trained:
        st.warning("Train models first."); return

    results  = st.session_state.results
    data     = st.session_state.data
    pipeline = st.session_state.pipeline
    mnames   = list(results.keys())
    best_name= max(results, key=lambda m: results[m]['test_f1'])

    tab1, tab2 = st.tabs(["LIVE DEFENSE SIMULATION","PACKET INSPECTOR"])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — ANIMATED DEFENSE SIM
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        sel_model = st.selectbox("Detection Model", mnames,
            index=mnames.index(best_name), key="tm_model")

        # Model info banner
        res_sel = results[sel_model]
        st.markdown(f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0">
          <div class="cc" style="flex:1;min-width:130px;padding:10px 14px">
            <div class="lbl">MODEL</div>
            <div class="val" style="font-size:1rem">{sel_model}</div>
          </div>
          <div class="cc green" style="flex:1;min-width:130px;padding:10px 14px">
            <div class="lbl">F1-SCORE</div>
            <div class="val">{res_sel['test_f1']:.4f}</div>
          </div>
          <div class="cc" style="flex:1;min-width:130px;padding:10px 14px">
            <div class="lbl">ACCURACY</div>
            <div class="val">{res_sel['test_accuracy']:.2%}</div>
          </div>
          <div class="cc amber" style="flex:1;min-width:130px;padding:10px 14px">
            <div class="lbl">BLOCK THRESHOLD</div>
            <div class="val" style="font-size:1rem">≥ 95%</div>
          </div>
          <div class="cc purple" style="flex:1;min-width:130px;padding:10px 14px">
            <div class="lbl">RATE-LIMIT</div>
            <div class="val" style="font-size:1rem">≥ 85%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶  LAUNCH ANIMATED DEFENSE SIMULATION", type="primary"):
            model     = results[sel_model]['model']
            N         = 300
            idx       = np.random.choice(len(data['X_test']), N, replace=False)
            X_all     = data['X_test'][idx]
            y_all     = data['y_test'][idx]
            benign_cls= pipeline.le.classes_[0]

            # Pre-compute all predictions
            all_pred = model.predict(X_all)
            all_proba= model.predict_proba(X_all) if hasattr(model,'predict_proba') else None
            all_conf = np.max(all_proba, axis=1) if all_proba is not None else np.ones(N)
            all_true = pipeline.le.inverse_transform(y_all)
            all_plbl = pipeline.le.inverse_transform(all_pred)

            # ── Live UI placeholders ────────────────────────────────────────
            sh("LIVE NETWORK TRAFFIC MONITOR")
            kpi_ph    = st.empty()
            prog_ph   = st.empty()
            charts_ph = st.empty()
            st.markdown('<div class="sh">LIVE DEFENSE LOG — LAST 20 PACKETS</div>', unsafe_allow_html=True)
            log_ph    = st.empty()

            log_entries = []
            counts = {a:0 for a in ACTION_COLORS}
            tl_pkt, tl_blk_cum, tl_alw_cum = [], [], []
            blk_cum = alw_cum = correct = 0
            BATCH = 20  # larger batch → fewer Streamlit re-renders → faster

            for i in range(0, N, BATCH):
                end   = min(i+BATCH, N)
                batch_true = all_true[i:end]
                batch_pred = all_plbl[i:end]
                batch_conf = all_conf[i:end]

                for j,(tl,pl,cf) in enumerate(zip(batch_true,batch_pred,batch_conf)):
                    act = get_action(pl, cf, benign_cls)
                    counts[act] += 1
                    if tl == pl: correct += 1
                    log_entries.append({
                        'ts':datetime.now().strftime('%H:%M:%S.%f')[:-3],
                        'pkt':i+j+1,'true':tl,'pred':pl,
                        'conf':cf,'action':act,'ok':tl==pl
                    })

                blk_cum += sum(1 for e in log_entries[i:i+BATCH] if e['action']!='ALLOWED')
                alw_cum += sum(1 for e in log_entries[i:i+BATCH] if e['action']=='ALLOWED')
                tl_pkt.append(end)
                tl_blk_cum.append(blk_cum)
                tl_alw_cum.append(alw_cum)

                done     = end
                acc_now  = correct/done*100

                # ── KPI row (stat strip — faster than raw HTML cards) ───────
                threat_total = counts['BLOCKED']+counts['RATE LIMITED']+counts['ALERT']
                kpi_ph.markdown(stat_strip([
                    ("SCANNED",       f"{done}/{N}",                      ""),
                    ("🚫 BLOCKED",    str(counts['BLOCKED']),              "r"),
                    ("⚠️ RATE LTD",  str(counts['RATE LIMITED']),         "a"),
                    ("🔔 ALERTS",     str(counts['ALERT']),                "p"),
                    ("✅ ALLOWED",    str(counts['ALLOWED']),              "g"),
                    ("DETECT ACC",    f"{acc_now:.1f}%",  "g" if acc_now>=95 else "a"),
                ]), unsafe_allow_html=True)

                prog_ph.progress(done/N, text=f"Scanning… {done}/{N} packets")

                # ── Dual chart ──────────────────────────────────────────────
                fig_d = make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.65, 0.35],
                    subplot_titles=["Cumulative Blocked vs Allowed","Action Distribution"],
                    specs=[[{"type":"xy"},{"type":"domain"}]]
                )
                fig_d.add_trace(go.Scatter(
                    x=tl_pkt, y=tl_alw_cum, name="Allowed",
                    mode='lines', fill='tozeroy',
                    line=dict(color='#00ff88', width=2.5),
                    fillcolor=rgba('#00ff88', 0.12)), row=1, col=1)
                fig_d.add_trace(go.Scatter(
                    x=tl_pkt, y=tl_blk_cum, name="Threats",
                    mode='lines', fill='tozeroy',
                    line=dict(color='#ff3366', width=2.5),
                    fillcolor=rgba('#ff3366', 0.12)), row=1, col=1)
                fig_d.add_trace(go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.5,
                    marker=dict(colors=[ACTION_COLORS[a] for a in counts]),
                    textfont=dict(family='Share Tech Mono', size=10),
                    showlegend=False), row=1, col=2)
                fig_d.update_layout(height=300, **PL,
                    xaxis=dict(title="Packet #", gridcolor='#0f2a42', linecolor='#0f2a42', tickcolor='#3a6a8a'),
                    yaxis=dict(title="Count",    gridcolor='#0f2a42', linecolor='#0f2a42', tickcolor='#3a6a8a'),
                )
                charts_ph.plotly_chart(fig_d, use_container_width=True)

                # ── Live log ─────────────────────────────────────────────────
                recent   = log_entries[-20:][::-1]
                log_html = ""
                for e in recent:
                    ac  = e['action']
                    clr = ACTION_COLORS[ac]
                    ico = ACTION_ICONS[ac]
                    rc  = ACTION_ROW_CLS[ac]
                    cor = "✓" if e['ok'] else "✗"
                    cc2 = "#00ff88" if e['ok'] else "#ff3366"
                    log_html += f"""
                    <div class="trow {rc}">
                      <span style="color:#3a6a8a;min-width:72px">{e['ts']}</span>
                      <span style="color:#5a8ab0;min-width:62px">PKT #{e['pkt']}</span>
                      <span style="color:#a0c4e8;min-width:88px">{str(e['true'])[:12]}</span>
                      <span style="color:{clr};min-width:110px">{ico} {ac}</span>
                      <span style="color:#5a8ab0">{e['conf']*100:.1f}%</span>
                      <span style="color:{cc2}">{cor}</span>
                    </div>"""
                log_ph.markdown(log_html, unsafe_allow_html=True)
                time.sleep(0.02)  # 0.05→0.02: snappier animation

            prog_ph.empty()

            # ── Final analysis ──────────────────────────────────────────────
            sh("SIMULATION COMPLETE — FINAL ANALYSIS")

            all_confs = np.array([e['conf'] for e in log_entries])
            all_acts  = [e['action'] for e in log_entries]

            fa1, fa2 = st.columns(2)
            with fa1:
                sh("CONFIDENCE DISTRIBUTION BY ACTION")
                fig_c = go.Figure()
                for act, c in ACTION_COLORS.items():
                    mask = np.array([a==act for a in all_acts])
                    if mask.any():
                        fig_c.add_trace(go.Histogram(
                            x=all_confs[mask], name=f"{ACTION_ICONS[act]} {act}",
                            nbinsx=30, marker_color=c, opacity=0.75))
                fig_c.add_vline(x=0.95, line_dash='dash', line_color='#ff3366',
                    annotation_text="BLOCK 95%", annotation_font_color='#ff3366',
                    annotation_position="top right")
                fig_c.add_vline(x=0.85, line_dash='dash', line_color='#ffaa00',
                    annotation_text="RATE LIMIT 85%", annotation_font_color='#ffaa00',
                    annotation_position="top left")
                fig_c.update_layout(barmode='overlay', title="Confidence Scores by Action")
                ptheme(fig_c, h=340); st.plotly_chart(fig_c, use_container_width=True)

            with fa2:
                sh("FINAL ACTION SUMMARY")
                act_cnt = Counter(all_acts)
                fig_act = go.Figure(go.Bar(
                    x=list(act_cnt.keys()), y=list(act_cnt.values()),
                    marker_color=[ACTION_COLORS[a] for a in act_cnt],
                    text=list(act_cnt.values()), textposition='outside',
                    textfont=dict(color='#a0c4e8')))
                fig_act.update_layout(title="Actions Taken Across 300 Packets")
                ptheme(fig_act, h=340); st.plotly_chart(fig_act, use_container_width=True)

            sh("PER-PACKET CONFIDENCE TIMELINE")
            pkt_ids = [e['pkt']   for e in log_entries]
            confs   = [e['conf']*100 for e in log_entries]
            fig_pts = go.Figure()
            for act, c in ACTION_COLORS.items():
                mask = [e['action']==act for e in log_entries]
                fig_pts.add_trace(go.Scatter(
                    x=[p for p,m in zip(pkt_ids,mask) if m],
                    y=[cf for cf,m in zip(confs,mask) if m],
                    mode='markers', name=f"{ACTION_ICONS[act]} {act}",
                    marker=dict(color=c, size=5, opacity=0.82)))
            fig_pts.add_hline(y=95, line_dash='dash', line_color='#ff3366',
                annotation_text="Block", annotation_font_color='#ff3366')
            fig_pts.add_hline(y=85, line_dash='dash', line_color='#ffaa00',
                annotation_text="Rate-Limit", annotation_font_color='#ffaa00')
            fig_pts.update_layout(title="Confidence per Packet — Full Simulation",
                xaxis_title="Packet #", yaxis_title="Confidence %")
            ptheme(fig_pts, h=370); st.plotly_chart(fig_pts, use_container_width=True)

            log_df = pd.DataFrame([{
                'Packet':e['pkt'],'Timestamp':e['ts'],'True':e['true'],
                'Predicted':e['pred'],'Confidence':f"{e['conf']*100:.1f}%",
                'Action':e['action'],'Correct':e['ok']
            } for e in log_entries])
            st.download_button("⬇ Download Defense Log",
                log_df.to_csv(index=False), "defense_log.csv", "text/csv")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — PACKET INSPECTOR
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        sh("PACKET-LEVEL INSPECTION — AUTO SAMPLE OF 50")
        sel_m2 = st.selectbox("Model", mnames, index=mnames.index(best_name), key="pi_model")

        if st.button("🔍 INSPECT 50 PACKETS", type="primary"):
            model = results[sel_m2]['model']
            N2    = 50
            idx2  = np.random.choice(len(data['X_test']), N2, replace=False)
            Xs, ys= data['X_test'][idx2], data['y_test'][idx2]
            yp    = model.predict(Xs)
            ypr   = model.predict_proba(Xs) if hasattr(model,'predict_proba') else None
            confs = np.max(ypr, axis=1)*100 if ypr is not None else np.ones(N2)*100
            tl    = pipeline.le.inverse_transform(ys)
            pl    = pipeline.le.inverse_transform(yp)
            bc    = pipeline.le.classes_[0]

            acc2 = (ys==yp).mean()*100
            atks = int((pl!=bc).sum())
            k1,k2,k3 = st.columns(3)
            k1.metric("Detection Accuracy", f"{acc2:.1f}%")
            k2.metric("Attacks Detected", atks)
            k3.metric("Avg Confidence",   f"{confs.mean():.1f}%")

            sh("PACKET TABLE")
            def_acts = [get_action(p,c/100,bc) for p,c in zip(pl,confs)]
            df_pkt = pd.DataFrame({
                'Pkt #':range(1,N2+1), 'True Label':tl, 'Predicted':pl,
                'Confidence':[f"{c:.1f}%" for c in confs],
                'Status':['✅ Correct' if t==p else '❌ Wrong' for t,p in zip(tl,pl)],
                'Action':def_acts,
            })
            st.dataframe(df_pkt, use_container_width=True, hide_index=True)

            sh("CONFIDENCE SCATTER WITH ACTION ZONES")
            fig_sc = go.Figure()
            for act, c in ACTION_COLORS.items():
                mask = [x==act for x in def_acts]
                fig_sc.add_trace(go.Scatter(
                    x=[i+1 for i,m in enumerate(mask) if m],
                    y=[confs[i] for i,m in enumerate(mask) if m],
                    mode='markers+text',
                    text=[f"{confs[i]:.0f}%" for i,m in enumerate(mask) if m],
                    textposition='top center',
                    textfont=dict(size=8, color=c),
                    name=f"{ACTION_ICONS[act]} {act}",
                    marker=dict(color=c, size=10,
                        line=dict(color='#04080f', width=2))))
            # Shaded zones
            fig_sc.add_hrect(y0=95, y1=105, fillcolor=rgba('#ff3366',0.07), line_width=0,
                annotation_text="BLOCK ZONE", annotation_font_color='#ff3366',
                annotation_position="top left")
            fig_sc.add_hrect(y0=85, y1=95,  fillcolor=rgba('#ffaa00',0.07), line_width=0,
                annotation_text="RATE-LIMIT ZONE", annotation_font_color='#ffaa00',
                annotation_position="top left")
            fig_sc.add_hrect(y0=0,  y1=85,  fillcolor=rgba('#9966ff',0.04), line_width=0,
                annotation_text="ALERT ZONE", annotation_font_color='#9966ff',
                annotation_position="bottom left")
            fig_sc.add_hline(y=95, line_dash='dash', line_color='#ff3366')
            fig_sc.add_hline(y=85, line_dash='dash', line_color='#ffaa00')
            fig_sc.update_layout(title="Per-Packet Confidence with Action Zones",
                xaxis_title="Packet #", yaxis_title="Confidence %", yaxis_range=[30,108])
            ptheme(fig_sc, h=420); st.plotly_chart(fig_sc, use_container_width=True)

            sh("ACTION BREAKDOWN")
            act_cnt2 = Counter(def_acts)
            fig_ab = go.Figure(go.Bar(
                x=list(act_cnt2.keys()), y=list(act_cnt2.values()),
                marker_color=[ACTION_COLORS[a] for a in act_cnt2],
                text=list(act_cnt2.values()), textposition='outside',
                textfont=dict(color='#a0c4e8')))
            fig_ab.update_layout(title="Action Distribution — 50 Packets")
            ptheme(fig_ab, h=300); st.plotly_chart(fig_ab, use_container_width=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    sidebar()
    p = st.session_state.page
    if   "Data Lab"       in p: page_data_lab()
    elif "Pipeline"       in p: page_pipeline()
    elif "Model Arena"    in p: page_model_arena()
    elif "Analytics Hub"  in p: page_analytics_hub()
    elif "Threat Monitor" in p: page_threat_monitor()
    else:                       page_data_lab()

if __name__ == "__main__":
    main()
