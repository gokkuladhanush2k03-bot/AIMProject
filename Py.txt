import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dark Theme ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    h1, h2, h3, h4, h5, h6 { color: #64ffda !important; }
    .stMetric { background-color: #1a1a2e; border-radius: 10px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
    .stButton>button { background-color: #0f3460; color: white; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Earnings Manipulation Dashboard", layout="wide")
st.title("💰 Earnings Manipulation Detection Dashboard")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])
if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)
st.success("✅ File loaded successfully!")

required_cols = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'ACCR', 'LEVI', 'Manipulator']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# --- Class Distribution Only ---
st.subheader("📊 Class Distribution")
class_dist = df['Manipulator'].value_counts()
st.bar_chart(class_dist)

# Prepare data
X = df[['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'ACCR', 'LEVI']]
y = df['Manipulator'].map({'No': 0, 'Yes': 1})

# Sidebar
st.sidebar.header("⚙️ Model Settings")
run_mode = st.sidebar.radio("Run Mode", ["Single Model", "All Models"])
model_choice = None
if run_mode == "Single Model":
    model_choice = st.sidebar.selectbox("Model", ["SVM", "KNN", "Naive Bayes", "AdaBoost", "XGBoost"])

# --- Test Split Selection ---
st.sidebar.subheader("SplitOptions")
test_split_options = [0.30, 0.25, 0.20]
selected_splits = st.sidebar.multiselect(
    "Select Test Splits",
    options=test_split_options,
    default=test_split_options  # default: all
)
enable_tuning = st.sidebar.checkbox("Enable Tuning", value=True)
run_button = st.sidebar.button("🚀 Run")

if not selected_splits:
    st.sidebar.warning("⚠️ Please select at least one test split.")
    st.stop()

# Generate splits only for selected test sizes
all_splits = []
scaler = StandardScaler()
for ts in selected_splits:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    all_splits.append({
        'test_size': ts,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled
    })

def evaluate(model_name, y_true, y_pred, y_prob, cv_mean=None, cv_std=None):
    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
    if cv_mean is not None:
        result["CV Mean ROC-AUC"] = cv_mean
        result["CV Std ROC-AUC"] = cv_std
    return result

if run_button:
    results_list = []

    def plot_curves(y_true, y_prob, model_name):
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {model_name}')
        ax1.legend(loc="lower right")
        st.pyplot(fig1)
        plt.close(fig1)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(recall, precision, color='teal', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {model_name}')
        ax2.legend(loc="lower left")
        st.pyplot(fig2)
        plt.close(fig2)

    # --- Run Models ---
    model_order = ["SVM", "KNN", "Naive Bayes", "AdaBoost", "XGBoost"]

    for model_type in model_order:
        if run_mode == "All Models" or model_choice == model_type:
            if model_type == "SVM":
                param_grid = {'C': [0.1,1,10,100], 'kernel': ['linear','rbf'], 'gamma': ['scale','auto',0.1,1]}
                for split in all_splits:
                    svm = SVC(probability=True, random_state=42)
                    if enable_tuning:
                        grid = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', refit=True, n_jobs=-1)
                        grid.fit(split['X_train_scaled'], split['y_train'])
                        model, params = grid.best_estimator_, grid.best_params_
                        cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
                        cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
                    else:
                        svm.fit(split['X_train_scaled'], split['y_train'])
                        model, params, cv_mean, cv_std = svm, "Default", None, None
                    y_pred = model.predict(split['X_test_scaled'])
                    y_prob = model.predict_proba(split['X_test_scaled'])[:,1]
                    metrics = evaluate(f"SVM{'_tuned' if enable_tuning else ''}", split['y_test'], y_pred, y_prob, cv_mean, cv_std)
                    metrics.update({'Best Parameters': params, 'Test Size': split['test_size']})
                    results_list.append(metrics)
            elif model_type == "KNN":
                param_grid = {'n_neighbors':[3,5,7,9,11], 'weights':['uniform','distance'], 'metric':['euclidean','manhattan','minkowski']}
                for split in all_splits:
                    knn = KNeighborsClassifier()
                    if enable_tuning:
                        grid = GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc', refit=True, n_jobs=-1)
                        grid.fit(split['X_train_scaled'], split['y_train'])
                        model, params = grid.best_estimator_, grid.best_params_
                        cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
                        cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
                    else:
                        knn.fit(split['X_train_scaled'], split['y_train'])
                        model, params, cv_mean, cv_std = knn, "Default", None, None
                    y_pred = model.predict(split['X_test_scaled'])
                    y_prob = model.predict_proba(split['X_test_scaled'])[:,1]
                    metrics = evaluate(f"KNN{'_tuned' if enable_tuning else ''}", split['y_test'], y_pred, y_prob, cv_mean, cv_std)
                    metrics.update({'Best Parameters': params, 'Test Size': split['test_size']})
                    results_list.append(metrics)
            elif model_type == "Naive Bayes":
                param_grid = {'var_smoothing': np.logspace(0, -9, num=10)}
                for split in all_splits:
                    nb = GaussianNB()
                    if enable_tuning:
                        grid = GridSearchCV(nb, param_grid, cv=5, scoring='roc_auc', refit=True, n_jobs=-1)
                        grid.fit(split['X_train'], split['y_train'])
                        model, params = grid.best_estimator_, grid.best_params_
                        cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
                        cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
                    else:
                        nb.fit(split['X_train'], split['y_train'])
                        model, params, cv_mean, cv_std = nb, "Default", None, None
                    y_pred = model.predict(split['X_test'])
                    y_prob = model.predict_proba(split['X_test'])[:,1]
                    metrics = evaluate(f"Naive Bayes{'_tuned' if enable_tuning else ''}", split['y_test'], y_pred, y_prob, cv_mean, cv_std)
                    metrics.update({'Best Parameters': params, 'Test Size': split['test_size']})
                    results_list.append(metrics)
            elif model_type == "AdaBoost":
                param_grid = {'n_estimators':[50,100,200], 'learning_rate':[0.01,0.05,0.1,0.5]}
                for split in all_splits:
                    ada = AdaBoostClassifier(random_state=42)
                    if enable_tuning:
                        grid = GridSearchCV(ada, param_grid, cv=5, scoring='roc_auc', refit=True, n_jobs=-1)
                        grid.fit(split['X_train'], split['y_train'])
                        model, params = grid.best_estimator_, grid.best_params_
                        cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
                        cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
                    else:
                        ada.fit(split['X_train'], split['y_train'])
                        model, params, cv_mean, cv_std = ada, "Default", None, None
                    y_pred = model.predict(split['X_test'])
                    y_prob = model.predict_proba(split['X_test'])[:,1]
                    metrics = evaluate(f"AdaBoost{'_tuned' if enable_tuning else ''}", split['y_test'], y_pred, y_prob, cv_mean, cv_std)
                    metrics.update({'Best Parameters': params, 'Test Size': split['test_size']})
                    results_list.append(metrics)
            elif model_type == "XGBoost":
                param_grid = {
                    'n_estimators':[100,200,300],
                    'learning_rate':[0.01,0.05,0.1],
                    'max_depth':[3,4,5],
                    'subsample':[0.8,0.9,1.0],
                    'colsample_bytree':[0.8,0.9,1.0]
                }
                for split in all_splits:
                    xgb = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
                    if enable_tuning:
                        grid = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', refit=True, n_jobs=-1)
                        grid.fit(split['X_train'], split['y_train'])
                        model, params = grid.best_estimator_, grid.best_params_
                        cv_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
                        cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
                    else:
                        xgb.fit(split['X_train'], split['y_train'])
                        model, params, cv_mean, cv_std = xgb, "Default", None, None
                    y_pred = model.predict(split['X_test'])
                    y_prob = model.predict_proba(split['X_test'])[:,1]
                    metrics = evaluate(f"XGBoost{'_tuned' if enable_tuning else ''}", split['y_test'], y_pred, y_prob, cv_mean, cv_std)
                    metrics.update({'Best Parameters': params, 'Test Size': split['test_size']})
                    results_list.append(metrics)

    # --- Beneish Baseline ---
    df_beneish = df.copy()
    df_beneish['M_Score'] = (
        -4.84 +
        0.92*(df_beneish['DSRI'] - 1) +
        0.528*(1 - df_beneish['GMI']) +
        0.404*(1 - df_beneish['AQI']) +
        0.892*(df_beneish['SGI'] - 1) +
        0.115*(1 - df_beneish['DEPI']) -
        0.172*(df_beneish['SGAI'] - 1) +
        4.679*df_beneish['ACCR'] -
        0.327*(df_beneish['LEVI'] - 1)
    )
    y_true = df_beneish['Manipulator'].map({'No': 0, 'Yes': 1})
    y_pred = (df_beneish['M_Score'] > -1.78).astype(int)
    beneish_metrics = evaluate("Beneish (rule)", y_true, y_pred, df_beneish['M_Score'])
    beneish_metrics.update({
        'Best Parameters': 'N/A',
        'Test Size': 'Full',
        'CV Mean ROC-AUC': np.nan,
        'CV Std ROC-AUC': np.nan
    })
    results_list.append(beneish_metrics)

    # --- Display Results ---
    results_df = pd.DataFrame(results_list)

    col_order = [
    "Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC",
    "CV Mean ROC-AUC", "CV Std ROC-AUC", "Test Size", "Best Parameters"
]
    results_df = results_df[col_order]

    st.subheader("📊 Full Model Results")

    def safe_format(x):
        return f"{x:.4f}" if pd.notna(x) else "N/A"

    # ✅ FIXED: Removed gmap
    styled_df = results_df.style.format({
        col: safe_format for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "CV Mean ROC-AUC", "CV Std ROC-AUC"]
    }).background_gradient(
        cmap='Blues',
        subset=["Accuracy", "ROC-AUC", "CV Mean ROC-AUC"]
    )

    st.dataframe(styled_df)

    # --- Best per model type ---
    best_models = []
    for model_type in results_df['Model'].unique():
        subset = results_df[results_df['Model'] == model_type]
        best_row = subset.loc[subset['Accuracy'].idxmax()]
        best_models.append(best_row.to_dict())
    best_df = pd.DataFrame(best_models)

    st.subheader("🏆 Best Model per Type")
    # ✅ FIXED: Removed gmap
    styled_best = best_df[col_order].style.format({
        col: safe_format for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "CV Mean ROC-AUC", "CV Std ROC-AUC"]
    }).background_gradient(
        cmap='Greens',
        subset=["Accuracy", "ROC-AUC", "CV Mean ROC-AUC"]
    )
    st.dataframe(styled_best)

    # --- Overall best ---
    best_overall = best_df.loc[best_df['Accuracy'].idxmax()]
    tie = best_df[best_df['Accuracy'] == best_overall['Accuracy']]
    if len(tie) > 1:
        best_overall = tie.loc[tie['ROC-AUC'].idxmax()]
    st.success(f"🥇 **Overall Best Model**: `{best_overall['Model']}` | Accuracy: {best_overall['Accuracy']:.4f} | ROC-AUC: {best_overall['ROC-AUC']:.4f}")

    # --- Detailed Analysis for Best Model ---
    st.subheader("🔍 Detailed Analysis for Best Model")
    best_model_name = best_overall['Model']
    best_ts = best_overall['Test Size']

    for split in all_splits:
        if split['test_size'] == best_ts:
            if 'SVM' in best_model_name or 'KNN' in best_model_name:
                X_train_use, X_test_use = split['X_train_scaled'], split['X_test_scaled']
            else:
                X_train_use, X_test_use = split['X_train'], split['X_test']
            y_train_use, y_test_use = split['y_train'], split['y_test']
            break

    best_params_row = results_df[(results_df['Model'] == best_model_name) & (results_df['Test Size'] == best_ts)].iloc[0]
    best_params = best_params_row['Best Parameters']

    if 'SVM' in best_model_name:
        model_cls = SVC(probability=True, random_state=42)
    elif 'KNN' in best_model_name:
        model_cls = KNeighborsClassifier()
    elif 'Naive Bayes' in best_model_name:
        model_cls = GaussianNB()
    elif 'AdaBoost' in best_model_name:
        model_cls = AdaBoostClassifier(random_state=42)
    elif 'XGBoost' in best_model_name:
        model_cls = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)

    if enable_tuning and isinstance(best_params, dict):
        best_mod = model_cls.set_params(**best_params)
    else:
        best_mod = model_cls

    if 'SVM' in best_model_name or 'KNN' in best_model_name:
        scaler_temp = StandardScaler()
        X_train_use = scaler_temp.fit_transform(X_train_use)
        X_test_use = scaler_temp.transform(X_test_use)

    best_mod.fit(X_train_use, y_train_use)
    y_pred_best = best_mod.predict(X_test_use)
    y_prob_best = best_mod.predict_proba(X_test_use)[:, 1]

    # Confusion Matrix
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test_use, y_pred_best)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    st.pyplot(fig)
    plt.close()

    # Classification Report
    st.write("**Classification Report**")
    report = classification_report(y_test_use, y_pred_best, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

    # Curves
    plot_curves(y_test_use, y_prob_best, best_model_name)

    # Download
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Results", csv, "earnings_manipulation_results.csv", "text/csv")

# --- Correlation Matrix ---
st.subheader("📈 Correlation Matrix")
df_corr = df.copy()
df_corr['Manipulator'] = df_corr['Manipulator'].map({'No': 0, 'Yes': 1})
numeric_df = df_corr.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(cax)
    st.pyplot(fig)

st.markdown("---")
st.caption("© 2026 Earnings Manipulation Detection Dashboard")