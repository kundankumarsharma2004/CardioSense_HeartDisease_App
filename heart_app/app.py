"""
Heart Disease Prediction System — Flask Backend
Models: Logistic Regression | Random Forest | SVM | KNN
"""

import warnings, os, io, base64, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify, send_file
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# ─── Palette ─────────────────────────────────────────
COLORS   = ['#2563EB', '#10B981', '#F59E0B', '#EF4444']
MODEL_COLORS = {
    'Logistic Regression': '#2563EB',
    'Random Forest':       '#10B981',
    'SVM':                 '#F59E0B',
    'KNN':                 '#EF4444',
}

# ─── Build / train models once at startup ────────────
def build_dataset(n=1025, seed=42):
    np.random.seed(seed)
    age      = np.random.randint(29, 78, n)
    sex      = np.random.randint(0, 2, n)
    cp       = np.random.randint(0, 4, n)
    trestbps = np.random.randint(94, 200, n)
    chol     = np.random.randint(126, 565, n)
    fbs      = np.random.randint(0, 2, n)
    restecg  = np.random.randint(0, 3, n)
    thalach  = np.random.randint(71, 202, n)
    exang    = np.random.randint(0, 2, n)
    oldpeak  = np.round(np.random.uniform(0, 6.2, n), 1)
    slope    = np.random.randint(0, 3, n)
    ca       = np.random.randint(0, 5, n)
    thal     = np.random.randint(0, 4, n)

    risk = (
        (age > 55).astype(float)      * 0.30 +
        (sex == 1).astype(float)      * 0.15 +
        (cp  == 0).astype(float)      * 0.20 +
        (thalach < 140).astype(float) * 0.15 +
        (exang == 1).astype(float)    * 0.20 +
        (oldpeak > 2).astype(float)   * 0.15 +
        (ca > 0).astype(float)        * 0.15 +
        np.random.normal(0, 0.1, n)
    )
    target = (risk > 0.55).astype(int)

    return pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal, 'target': target
    })


FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']

df = build_dataset()
X  = df[FEATURES]
y  = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

MODELS = {
    'Logistic Regression': Pipeline([
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('sc', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'SVM': Pipeline([
        ('sc', StandardScaler()),
        ('clf', SVC(probability=True, kernel='rbf', random_state=42))
    ]),
    'KNN': Pipeline([
        ('sc', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
RESULTS = {}
for name, pipe in MODELS.items():
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv_sc   = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy').mean()
    cm      = confusion_matrix(y_test, y_pred)
    cr      = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    RESULTS[name] = {
        'accuracy': acc, 'auc': auc, 'cv_score': cv_sc,
        'y_pred': y_pred.tolist(), 'y_proba': y_proba.tolist(),
        'cm': cm.tolist(), 'report': cr,
        'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
        'pipe': pipe
    }

# Feature importance from RF
rf_clf = RESULTS['Random Forest']['pipe'].named_steps['clf']
FEATURE_IMP = dict(zip(FEATURES, rf_clf.feature_importances_.tolist()))

# Dataset stats
DATASET_STATS = {
    'total': len(df),
    'disease': int(df['target'].sum()),
    'no_disease': int((df['target'] == 0).sum()),
    'train_size': len(X_train),
    'test_size': len(X_test),
}

print("✅  All models trained successfully.")
for n, r in RESULTS.items():
    print(f"   {n:<22} Acc={r['accuracy']:.4f}  AUC={r['auc']:.4f}")


# ─── Helper: figure → base64 PNG ─────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='white')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─── Routes ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stats')
def api_stats():
    perf = {}
    for name, r in RESULTS.items():
        perf[name] = {
            'accuracy': round(r['accuracy'], 4),
            'auc':      round(r['auc'],      4),
            'cv_score': round(r['cv_score'], 4),
            'precision': round(r['report']['1']['precision'], 4),
            'recall':    round(r['report']['1']['recall'],    4),
            'f1':        round(r['report']['1']['f1-score'],  4),
        }
    return jsonify({
        'dataset': DATASET_STATS,
        'performance': perf,
        'feature_importance': FEATURE_IMP,
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    try:
        patient = pd.DataFrame([{f: float(data[f]) for f in FEATURES}])
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    predictions = {}
    for name, r in RESULTS.items():
        prob = float(r['pipe'].predict_proba(patient)[0][1])
        predictions[name] = {
            'probability': round(prob, 4),
            'risk': 'HIGH' if prob >= 0.5 else 'LOW',
            'color': MODEL_COLORS[name],
        }

    probs   = [v['probability'] for v in predictions.values()]
    avg_prob = round(sum(probs) / len(probs), 4)
    consensus = 'HIGH' if avg_prob >= 0.5 else 'LOW'
    high_votes = sum(1 for v in predictions.values() if v['risk'] == 'HIGH')

    return jsonify({
        'predictions': predictions,
        'avg_probability': avg_prob,
        'consensus': consensus,
        'high_votes': high_votes,
        'total_models': len(predictions),
    })


@app.route('/api/chart/roc')
def chart_roc():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    for i, (name, r) in enumerate(RESULTS.items()):
        ax.plot(r['fpr'], r['tpr'], color=COLORS[i], lw=2.2,
                label=f"{name}  (AUC = {r['auc']:.3f})")
    ax.plot([0,1],[0,1],'--', color='#CBD5E1', lw=1.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold', pad=14)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, color='#94A3B8')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return jsonify({'image': b64})


@app.route('/api/chart/confusion')
def chart_confusion():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.patch.set_facecolor('white')
    labels = ['No\nDisease', 'Disease']
    cmaps  = ['Blues','Greens','Oranges','Reds']
    for i, (name, r) in enumerate(RESULTS.items()):
        sns.heatmap(np.array(r['cm']), annot=True, fmt='d',
                    cmap=cmaps[i], ax=axes[i], cbar=False,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 14, 'weight': 'bold'})
        axes[i].set_title(name, fontweight='bold', fontsize=10, pad=8)
        axes[i].set_xlabel('Predicted', fontsize=9)
        axes[i].set_ylabel('Actual', fontsize=9)
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return jsonify({'image': b64})


@app.route('/api/chart/feature_importance')
def chart_feature_imp():
    fi = pd.Series(FEATURE_IMP).sort_values()
    feature_labels = {
        'age':'Age','sex':'Sex','cp':'Chest Pain',
        'trestbps':'Resting BP','chol':'Cholesterol',
        'fbs':'Fasting BS','restecg':'Rest ECG',
        'thalach':'Max HR','exang':'Exercise Angina',
        'oldpeak':'ST Depression','slope':'ST Slope',
        'ca':'Major Vessels','thal':'Thalassemia'
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    bars = ax.barh(
        [feature_labels.get(f, f) for f in fi.index],
        fi.values,
        color=[plt.cm.RdYlGn(v / fi.max()) for v in fi.values],
        edgecolor='white', height=0.65
    )
    for bar, val in zip(bars, fi.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8.5)
    ax.set_title('Feature Importance — Random Forest', fontsize=13,
                 fontweight='bold', pad=14)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, color='#94A3B8')
    ax.tick_params(left=False)
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return jsonify({'image': b64})


@app.route('/api/chart/prob_dist')
def chart_prob_dist():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor('white')
    for ax, (name, r), col in zip(axes.flatten(), RESULTS.items(), COLORS):
        proba  = np.array(r['y_proba'])
        y_true = np.array(y_test)
        ax.set_facecolor('#F8FAFC')
        ax.hist(proba[y_true==0], bins=22, alpha=0.65, color='#10B981',
                label='No Disease', edgecolor='white')
        ax.hist(proba[y_true==1], bins=22, alpha=0.65, color='#EF4444',
                label='Heart Disease', edgecolor='white')
        ax.axvline(0.5, color='#1E293B', lw=1.8, ls='--', label='Threshold 0.5')
        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted Probability', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=0.2)
    fig.suptitle('Probability Score Distributions', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return jsonify({'image': b64})


@app.route('/api/chart/accuracy_bar')
def chart_accuracy():
    names = list(RESULTS.keys())
    accs  = [RESULTS[n]['accuracy'] for n in names]
    aucs  = [RESULTS[n]['auc']      for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor('white')
    for ax, vals, title, ylabel in [
        (axes[0], accs, 'Model Accuracy', 'Accuracy'),
        (axes[1], aucs, 'ROC-AUC Score',  'AUC'),
    ]:
        ax.set_facecolor('#F8FAFC')
        bars = ax.bar(names, vals, color=COLORS, width=0.5,
                      edgecolor='white', linewidth=1.5)
        ax.set_ylim(0.6, 1.0)
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='x', rotation=12, labelsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3, color='#94A3B8')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    b64 = fig_to_b64(fig); plt.close(fig)
    return jsonify({'image': b64})


if __name__ == '__main__':
    app.run(debug=True, port=5050)
