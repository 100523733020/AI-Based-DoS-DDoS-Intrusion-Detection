"""
DoS & DDoS Multi-Class Detection — Hybrid CNN-BiLSTM + Attention
Run: python ddos.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Bidirectional, LSTM,
                                     Dense, Dropout, BatchNormalization,
                                     Flatten, Activation, Reshape)
from tensorflow.keras.utils import to_categorical

import shap, xgboost as xgb

np.random.seed(42)
tf.random.set_seed(42)

ATTACK_NAMES = ["Benign","SYN Flood","UDP Flood","HTTP Flood","DNS Amp",
                "NTP Amp","SNMP Amp","MSSQL","SSDP Amp","LDAP","TFTP","NetBIOS"]
N_CLASSES = len(ATTACK_NAMES)

# 1. DATA
print("Loading data...")
weights = [0.35,0.12,0.10,0.09,0.07,0.06,0.05,0.04,0.04,0.03,0.03,0.02]
X_parts, y_parts = [], []
for i, w in enumerate(weights):
    n = int(60000 * w)
    Xc, _ = make_classification(n_samples=n, n_features=78, n_informative=40,
                                 n_redundant=10, random_state=i)
    X_parts.append(Xc + np.random.randn(78) * i * 0.3)
    y_parts += [i] * n

X = np.vstack(X_parts)
y = np.array(y_parts)
idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]
print(f"Dataset: {X.shape}, Classes: {N_CLASSES}")

# 2. PREPROCESS
print("Preprocessing + SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"After SMOTE — Train: {X_train.shape}")

# 3. MODEL
def build_model(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    # Self-attention
    attn = Dense(1, activation='tanh')(x)
    attn = Flatten()(attn)
    attn = Activation('softmax')(attn)
    attn = Reshape((1, -1))(attn)
    x = keras.ops.matmul(attn, x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(n_cls, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("\nTraining CNN-BiLSTM model...")
model = build_model(X_train.shape[1], N_CLASSES)
model.summary()

X_tr_r = X_train.reshape(-1, X_train.shape[1], 1)
X_v_r  = X_val.reshape(-1, X_val.shape[1], 1)
X_te_r = X_test.reshape(-1, X_test.shape[1], 1)

history = model.fit(
    X_tr_r, to_categorical(y_train, N_CLASSES),
    validation_data=(X_v_r, to_categorical(y_val, N_CLASSES)),
    epochs=30, batch_size=512,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)],
    verbose=1
)

# 4. BASELINES
print("\nTraining baselines...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

xgb_clf = xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
xgb_clf.fit(X_train, y_train)

results = {
    "Random Forest": {"acc": accuracy_score(y_test, rf.predict(X_test)),
                      "f1":  f1_score(y_test, rf.predict(X_test), average='macro')},
    "XGBoost":       {"acc": accuracy_score(y_test, xgb_clf.predict(X_test)),
                      "f1":  f1_score(y_test, xgb_clf.predict(X_test), average='macro')},
}

y_pred = np.argmax(model.predict(X_te_r, verbose=0), axis=1)
results["CNN-BiLSTM\n(Proposed)"] = {
    "acc": accuracy_score(y_test, y_pred),
    "f1":  f1_score(y_test, y_pred, average='macro')
}

print("\n=== RESULTS ===")
for name, r in results.items():
    print(f"  {name.replace(chr(10),' '):<30} Acc: {r['acc']:.4f}  F1: {r['f1']:.4f}")

print("\n=== Classification Report (Proposed Model) ===")
print(classification_report(y_test, y_pred, target_names=ATTACK_NAMES))

# 5. PLOTS
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("DoS/DDoS Detection — CNN-BiLSTM with Self-Attention", fontsize=14, fontweight='bold')

axes[0,0].plot(history.history['accuracy'],     label='Train', color='#2E86AB', lw=2)
axes[0,0].plot(history.history['val_accuracy'], label='Val',   color='#C73E1D', lw=2)
axes[0,0].set_title('Training Accuracy'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(history.history['loss'],     label='Train', color='#2E86AB', lw=2)
axes[0,1].plot(history.history['val_loss'], label='Val',   color='#C73E1D', lw=2)
axes[0,1].set_title('Training Loss'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[a[:6] for a in ATTACK_NAMES],
            yticklabels=[a[:6] for a in ATTACK_NAMES], ax=axes[0,2])
axes[0,2].set_title('Confusion Matrix (Normalised)')
axes[0,2].tick_params(axis='x', rotation=45, labelsize=7)
axes[0,2].tick_params(axis='y', rotation=0,  labelsize=7)

names = list(results.keys())
accs  = [results[n]['acc'] for n in names]
f1s   = [results[n]['f1']  for n in names]
x = np.arange(len(names))
axes[1,0].bar(x - 0.2, accs, 0.4, label='Accuracy', color='#2E86AB', alpha=0.85)
axes[1,0].bar(x + 0.2, f1s,  0.4, label='Macro F1', color='#F18F01', alpha=0.85)
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(names, fontsize=8)
axes[1,0].set_ylim([0.8, 1.05]); axes[1,0].legend()
axes[1,0].set_title('Model Comparison'); axes[1,0].grid(axis='y', alpha=0.3)

report = classification_report(y_test, y_pred, target_names=ATTACK_NAMES, output_dict=True)
f1_per_class = [report[n]['f1-score'] for n in ATTACK_NAMES]
colors = plt.cm.tab20(np.linspace(0, 1, N_CLASSES))
axes[1,1].barh(ATTACK_NAMES, f1_per_class, color=colors)
axes[1,1].set_xlim([0, 1.1]); axes[1,1].set_title('Per-Class F1 Score')
axes[1,1].grid(axis='x', alpha=0.3)
for i, v in enumerate(f1_per_class):
    axes[1,1].text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)

print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X_test[:300])
mean_shap = np.mean([np.abs(s).mean(0) for s in shap_vals], axis=0)
top10 = np.argsort(mean_shap)[::-1][:10]
axes[1,2].barh([f"Feature {i}" for i in top10[::-1]], mean_shap[top10[::-1]], color='#7B2D8B', alpha=0.85)
axes[1,2].set_title('SHAP Feature Importance (Top-10)')
axes[1,2].set_xlabel('Mean |SHAP|'); axes[1,2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("ddos_results.png", dpi=150, bbox_inches='tight')
print("\nPlot saved -> ddos_results.png")
plt.show()