import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, RocCurveDisplay, confusion_matrix, matthews_corrcoef, average_precision_score, PrecisionRecallDisplay
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import joblib
import matplotlib
matplotlib.use('Agg')   # 使用无GUI后台，不调用tkinter，只保存图片，不弹出窗口
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import shap
import os

# =====================配置参数=====================
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MORGAN_RADIUS = 2
MORGAN_NBITS = 1024

# ==========新增全局字体设置【Times New Roman】==========
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300
matplotlib.rcParams['pdf.fonttype'] = 42  # pdf输出字体可编辑，期刊必备
matplotlib.rcParams['ps.fonttype'] = 42

# =====================特征列表（已删除Total_Score、N Natural Charge和pKa）=====================
feature_cols = [
    "HOMO-1", 
    "LUMO_HOMO_gap (eV)", 
    # "Total_Score",  # 已删除
    "Crash", 
    "Polar",
    "Molar Volume", 
    "O Natural Charge",
    # "N Natural Charge",  # 已删除
    "Tot(D)", 
    # "pKa"  # 已删除
]

smiles_col = "smi"
target_reg_col = "VX-hAChE"  #原始连续活性

# 二分类阈值策略：'median' / 'quantile25'
split_strategy = "median"

# 分类模型集合
models = {
    "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=100),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "XGBClassifier": xgb.XGBClassifier(random_state=RANDOM_SEED, n_estimators=100, objective="binary:logistic"),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "MLPClassifier": MLPClassifier(random_state=RANDOM_SEED, hidden_layer_sizes=(64,32), max_iter=2000)
}

def calc_morgan_fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(MORGAN_NBITS, dtype=np.float64)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return np.array(fp, dtype=np.float64)

# =====================读取数据 & 构造二分类标签=====================
df = pd.read_excel("data_file_LUMO_HOMO_gap_126_freeTotalScore_N_natural_charge_free_pKa.xlsx")
y_reg = df[target_reg_col].copy()

if split_strategy == "median":
    thresh = np.median(y_reg)
    y_class = (y_reg > thresh).astype(int)
elif split_strategy == "quantile25":
    thresh = np.quantile(y_reg, 0.75)
    y_class = (y_reg >= thresh).astype(int)
else:
    raise ValueError("split_strategy选择 median 或 quantile25")

print(f"二分类阈值 = {thresh:.3f}")
print(f"高活性样本(1): {np.sum(y_class)} 条；低活性样本(0): {len(y_class)-np.sum(y_class)} 条")

# 数值特征 + Morgan指纹拼接
X_num = df[feature_cols].copy()
fp_arr = np.array([calc_morgan_fp(s) for s in df[smiles_col]])
X_raw = np.hstack([X_num.values, fp_arr])
fp_names = [f"FP_{i}" for i in range(MORGAN_NBITS)]
all_feature_names = feature_cols + fp_names

print(f"\n样本数量：{len(df)}，物化特征数：{len(feature_cols)}，指纹维度：{MORGAN_NBITS}，总特征：{X_raw.shape[1]}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ===================== Applicability Domain (Tanimoto‑based AD) =====================
class ApplicabilityDomainTanimoto:
    def __init__(self, train_fingerprints, threshold=0.35):
        self.train_fps = train_fingerprints
        self.threshold = threshold

    def calc_max_tanimoto(self, mol_fp):
        sims = [TanimotoSimilarity(mol_fp, fp_tr) for fp_tr in self.train_fps]
        return max(sims) if len(self.train_fps) > 0 else 0.0

    def is_inside_AD(self, mol_fp):
        max_sim = self.calc_max_tanimoto(mol_fp)
        return {"max_tanimoto": max_sim, "inside_AD": bool(max_sim >= self.threshold)}

def get_rdkit_fp_obj(smi, radius, nBits):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)

# 仅一次划分索引，保证X与df样本严格对齐
df_idx = df.reset_index(drop=True).copy()
train_idx, test_idx = train_test_split(
    np.arange(len(df_idx)),
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_class.values
)

X_train = X_scaled[train_idx, :]
X_test = X_scaled[test_idx, :]
y_train = y_class.iloc[train_idx]
y_test = y_class.iloc[test_idx]

df_train_all = df_idx.iloc[train_idx].copy()
train_smiles = df_train_all[smiles_col].tolist()
train_fp_objs = [
    get_rdkit_fp_obj(s, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
    for s in train_smiles
    if get_rdkit_fp_obj(s, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS) is not None
]

ad_model = ApplicabilityDomainTanimoto(train_fp_objs, threshold=0.35)

# ✅保存RDKit原始指纹对象，供给外部虚拟筛选脚本使用
joblib.dump(train_fp_objs, "ad_train_fps.joblib")
print(f"\n==== Applicability Domain ====")
print(f"AD训练集有效指纹数量：{len(train_fp_objs)}")
print(f"AD训练指纹已保存 → ad_train_fps.joblib")

# ====================================================================================
# =====================模型训练与评估=====================
results = []
cv_strat = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for name, estimator in models.items():
    print(f"\n==== {name} ====")
    estimator.fit(X_train, y_train)
    y_pred_test = estimator.predict(X_test)
    y_pred_prob_test = estimator.predict_proba(X_test)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_pred_prob_test)
    test_prec = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    cv_auc = cross_val_score(estimator, X_train, y_train, cv=cv_strat, scoring="roc_auc")
    
    res_row = {
        "Model": name,
        "Test_AUC": round(test_auc, 4),
        "Test_Precision": round(test_prec, 4),
        "Test_Recall": round(test_recall, 4),
        "Test_F1": round(test_f1, 4),
        "CV_mean_AUC": round(np.mean(cv_auc), 4),
        "CV_std_AUC": round(np.std(cv_auc), 4)
    }
    results.append(res_row)
    print(res_row)

df_result = pd.DataFrame(results)
df_result.to_csv("binary_class_result.csv", index=False, encoding="utf-8-sig")
print("\n【二分类模型结果已保存 binary_class_result.csv】")
print(df_result.to_string(index=False))

# =====================绘制ROC曲线=====================
n_model = len(models)
ncol = 3
nrow = int(np.ceil(n_model / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 5 * nrow))
axes = axes.flatten()

for i, (name, estimator) in enumerate(models.items()):
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=axes[i], name=name)
    axes[i].plot([0, 1], [0, 1], "k--", alpha=0.6)
    axes[i].set_title(f"{name} ROC")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("binary_ROC_curves.png", dpi=300)
plt.close()
print("【ROC曲线图保存 binary_ROC_curves.png】")

# =====================树模型特征重要性输出=====================
tree_based = ["RandomForest", "GradientBoosting", "DecisionTree", "XGBClassifier"]
fi_list = []
for m_name in tree_based:
    est = models[m_name]
    fi = est.feature_importances_
    for feat, val in zip(all_feature_names, fi):
        fi_list.append({"Model": m_name, "Feature": feat, "Importance": round(val, 4)})

df_fi = pd.DataFrame(fi_list)
df_fi.to_csv("binary_tree_feature_importance.csv", index=False, encoding="utf-8-sig")
print("【树模型特征重要性保存 binary_tree_feature_importance.csv】")

# 保存RF模型与scaler
rf_model = models["RandomForest"]
joblib.dump(rf_model, "rf_binary_classifier.pkl")
joblib.dump(scaler, "standard_scaler.pkl")
print("Model saved: rf_binary_classifier.pkl , standard_scaler.pkl")

# ============ Y‑scrambling 置换检验 ============
def y_scramble_analysis(X_train, y_train, n_runs=50, seed=RANDOM_SEED):
    scramble_result = []
    cv_strat_scr = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    np.random.seed(seed)
    for run_idx in range(n_runs):
        y_shuffled = np.random.permutation(y_train)
        rf_scr = RandomForestClassifier(n_estimators=100, random_state=seed)
        cv_auc_scr = cross_val_score(rf_scr, X_train, y_shuffled, cv=cv_strat_scr, scoring="roc_auc")
        scramble_result.append({
            "run": run_idx + 1,
            "mean_cv_auc": np.mean(cv_auc_scr),
            "std_cv_auc": np.std(cv_auc_scr)
        })
    df_scramble = pd.DataFrame(scramble_result)
    df_scramble.to_csv("y_scrambling_result.csv", index=False, encoding="utf-8-sig")
    print("\n==== Y‑Scrambling置换检验完成 ====")
    print(f"置换次数：{n_runs}")
    print(f"打乱标签后平均CV‑AUC：{df_scramble['mean_cv_auc'].mean():.4f} ± {df_scramble['mean_cv_auc'].std():.4f}")
    print(f"50次置换最高CV‑AUC：{df_scramble['mean_cv_auc'].max():.4f}")
    return df_scramble

df_scram = y_scramble_analysis(X_train, y_train, n_runs=50, seed=RANDOM_SEED)

def plot_y_scrambling_hist(csv_path="y_scrambling_result.csv", real_model_cv_auc=0.5, savefig="y_scrambling_hist.png"):
    df_scr = pd.read_csv(csv_path)
    auc_list = df_scr["mean_cv_auc"].values
    plt.figure(figsize=(8, 5), dpi=300)
    plt.hist(auc_list, bins=15, alpha=0.7, color="#1f77b4", edgecolor="black", label=f"Y‑scrambling (n={len(auc_list)})")
    plt.axvline(x=0.5, color="gray", linestyle="--", label="Random expectation = 0.5")
    plt.axvline(x=real_model_cv_auc, color="red", lw=2.2, label=f"Real model CV‑AUC = {real_model_cv_auc:.2f}")
    plt.xlabel("CV‑AUC")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefig, dpi=300)
    plt.close()
    print(f"Y‑scrambling plot saved: {savefig}")
    print(f"Scrambled mean AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

# 自动读取RF真实CV‑AUC
rf_cv_auc = df_result.loc[df_result["Model"] == "RandomForest", "CV_mean_AUC"].values[0]
plot_y_scrambling_hist(csv_path="y_scrambling_result.csv", real_model_cv_auc=rf_cv_auc, savefig="y_scrambling_hist.png")

# ====================== 新增：训练集指标、MCC、混淆矩阵 ======================
results_ext = []
cv_strat_ext = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for name, estimator in models.items():
    # 训练集预测
    y_pred_train = estimator.predict(X_train)
    y_predprob_train = estimator.predict_proba(X_train)[:,1]
    # 测试集预测
    y_pred_test = estimator.predict(X_test)
    y_predprob_test = estimator.predict_proba(X_test)[:,1]
    
    # train metrics
    train_auc = roc_auc_score(y_train, y_predprob_train)
    train_mcc = matthews_corrcoef(y_train, y_pred_train)
    # test metrics
    test_auc = roc_auc_score(y_test, y_predprob_test)
    test_mcc = matthews_corrcoef(y_test, y_pred_test)
    
    cv_auc = cross_val_score(estimator, X_train, y_train, cv=cv_strat_ext, scoring="roc_auc")
    
    res_row = {
        "Model": name,
        "Train_AUC": round(train_auc,4),
        "Train_MCC": round(train_mcc,4),
        "Test_AUC": round(test_auc,4),
        "Test_MCC": round(test_mcc,4),
        "CV_mean_AUC": round(np.mean(cv_auc),4),
        "CV_std_AUC": round(np.std(cv_auc),4),
    }
    results_ext.append(res_row)
    print(res_row)

df_result_ext = pd.DataFrame(results_ext)
df_result_ext.to_csv("binary_class_result_with_train_mcc.csv", index=False, encoding="utf-8-sig")

# 绘制RF混淆矩阵（训练集、测试集）
rf = models["RandomForest"]
y_pred_tr_rf = rf.predict(X_train)
y_pred_te_rf = rf.predict(X_test)
cm_train = confusion_matrix(y_train, y_pred_tr_rf)
cm_test = confusion_matrix(y_test, y_pred_te_rf)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4.5))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title("RF Confusion Matrix (Train set)")
ax1.set_xlabel("Predicted label")
ax1.set_ylabel("True label")
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_title("RF Confusion Matrix (Test set)")
ax2.set_xlabel("Predicted label")
ax2.set_ylabel("True label")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=300)
plt.close()
print("混淆矩阵图已保存 rf_confusion_matrix.png")

# 对test集合每一个smile计算max_tanimoto，标记inside_AD
test_smiles = df_idx.iloc[test_idx][smiles_col].tolist()
test_fp_objs = []
valid_test_mask = []
for s in test_smiles:
    fpobj = get_rdkit_fp_obj(s, MORGAN_RADIUS, MORGAN_NBITS)
    test_fp_objs.append(fpobj)
    valid_test_mask.append(fpobj is not None)

test_df = df_idx.iloc[test_idx].copy()
test_df = test_df.reset_index(drop=True)
test_df["valid_fp"] = valid_test_mask

max_tan_list = []
inside_ad_list = []
for fp in test_fp_objs:
    if fp is None:
        max_tan_list.append(np.nan)
        inside_ad_list.append(False)
    else:
        ad_out = ad_model.is_inside_AD(fp)
        max_tan_list.append(ad_out["max_tanimoto"])
        inside_ad_list.append(ad_out["inside_AD"])

test_df["max_tanimoto"] = max_tan_list
test_df["inside_AD"] = inside_ad_list
# RF预测结果
test_df["y_true"] = y_test.values
test_df["y_pred_rf"] = rf.predict(X_test)
test_df["y_prob_rf"] = rf.predict_proba(X_test)[:,1]
test_df.to_csv("testset_AD_analysis.csv", index=False, encoding="utf-8-sig")

# AD内性能、AD外性能
inside_df = test_df[test_df["inside_AD"]==True]
outside_df = test_df[test_df["inside_AD"]==False]

print("\n==== Applicability Domain Test‑set Performance ====")
print(f"Total test samples: {len(test_df)}")
print(f"In‑AD samples: {len(inside_df)}")
print(f"Out‑AD samples: {len(outside_df)}")

if len(inside_df) > 3:
    auc_in = roc_auc_score(inside_df["y_true"], inside_df["y_prob_rf"])
    mcc_in = matthews_corrcoef(inside_df["y_true"], inside_df["y_pred_rf"])
    print(f"Inside AD -> AUC:{auc_in:.4f}, MCC:{mcc_in:.4f}")
else:
    print(f"Inside‑AD样本过少({len(inside_df)}),跳过评估")

if len(outside_df) > 3:
    auc_out = roc_auc_score(outside_df["y_true"], outside_df["y_prob_rf"])
    mcc_out = matthews_corrcoef(outside_df["y_true"], outside_df["y_pred_rf"])
    print(f"Outside AD -> AUC:{auc_out:.4f}, MCC:{mcc_out:.4f}")
else:
    print(f"Outside‑AD样本过少({len(outside_df)}),跳过评估")

# 多随机种子稳定性
def run_single_seed(seed):
    tr_idx, te_idx = train_test_split(np.arange(len(df_idx)), test_size=TEST_SIZE, random_state=seed, stratify=y_class.values)
    Xtr = X_scaled[tr_idx,:]
    Xte = X_scaled[te_idx,:]
    ytr = y_class.iloc[tr_idx]
    yte = y_class.iloc[te_idx]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    rf_s = RandomForestClassifier(random_state=seed, n_estimators=100)
    rf_s.fit(Xtr,ytr)
    cv_auc_s = cross_val_score(rf_s, Xtr, ytr, cv=skf, scoring="roc_auc")
    te_prob = rf_s.predict_proba(Xte)[:,1]
    te_auc = roc_auc_score(yte, te_prob)
    return {"seed":seed, "cv_mean":np.mean(cv_auc_s), "cv_std":np.std(cv_auc_s), "test_auc":te_auc}

seed_list = [0,42,123,2025,999]
multi_seed_res = [run_single_seed(s) for s in seed_list]
df_multi_seed = pd.DataFrame(multi_seed_res)
df_multi_seed.to_csv("multi_seed_stability.csv", index=False, encoding="utf-8-sig")

print("\n==== Multiple random seed stability analysis ====")
print(df_multi_seed.to_string(index=False))
print(f"Mean CV_AUC ± std: {df_multi_seed['cv_mean'].mean():.4f} ± {df_multi_seed['cv_mean'].std():.4f}")
print(f"Mean Test_AUC ± std: {df_multi_seed['test_auc'].mean():.4f} ± {df_multi_seed['test_auc'].std():.4f}")

# RF网格搜索调参
rf_base = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100)
param_grid = {
    "max_depth":[3,5,8,None],
    "min_samples_split":[4,8,12],
    "min_samples_leaf":[2,4,6]
}
grid_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
grid_search = GridSearchCV(estimator=rf_base,
                           param_grid=param_grid,
                           cv=grid_cv,
                           scoring="roc_auc",
                           n_jobs=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print("\n==== GridSearch for RandomForest ====")
print("Best parameters:", grid_search.best_params_)
print("Best CV‑AUC from grid search:", grid_search.best_score_)

# 使用最优参数模型评估
yptr_best = best_rf.predict_proba(X_train)[:,1]
ypte_best = best_rf.predict_proba(X_test)[:,1]
print(f"Tuned RF Train AUC: {roc_auc_score(y_train,yptr_best):.4f}")
print(f"Tuned RF Test AUC: {roc_auc_score(y_test,ypte_best):.4f}")

joblib.dump(best_rf, "rf_tuned_best.pkl")
print("调优完成，最优RF保存 rf_tuned_best.pkl")

# ==============================================================
# ==========【新增模块1】PR‑AUC / PR曲线（最优调优RF） ==========
y_predprob_train_best = best_rf.predict_proba(X_train)[:, 1]
y_predprob_test_best = best_rf.predict_proba(X_test)[:, 1]
train_ap = average_precision_score(y_train, y_predprob_train_best)
test_ap = average_precision_score(y_test, y_predprob_test_best)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4.5), dpi=300)
plt.subplots_adjust(wspace=0.1)

# ROC
RocCurveDisplay.from_estimator(best_rf, X_train, y_train, ax=ax1, name=f"Train AUC={roc_auc_score(y_train,y_predprob_train_best):.3f}")
RocCurveDisplay.from_estimator(best_rf, X_test, y_test, ax=ax1, name=f"Test AUC={roc_auc_score(y_test,y_predprob_test_best):.3f}")
ax1.plot([0,1],[0,1],"k--",alpha=0.6)
ax1.set_title("ROC curve of the optimized RF model")
ax1.set_xlabel("False Positive Rate (Positive label: 1)")
ax1.set_ylabel("True Positive Rate (Positive label: 1)")

# PR‑Curve
PrecisionRecallDisplay.from_predictions(y_train, y_predprob_train_best, ax=ax2, name=f"Train AP={train_ap:.3f}")
PrecisionRecallDisplay.from_predictions(y_test, y_predprob_test_best, ax=ax2, name=f"Test AP={test_ap:.3f}")
ax2.set_title("Precision‑Recall curve of the optimized RF model")
ax2.set_xlabel("Recall (Positive label: 1)")
ax2.set_ylabel("Precision (Positive label: 1)")

plt.tight_layout()
plt.savefig("tuned_rf_roc_pr.png", dpi=300)
plt.close()
print("[1] Tuned‑RF ROC+PR图已保存 tuned_rf_roc_pr.png")

# ==============================================================
# ==========【新增模块2】多模型测试集性能对比条形图 ==========
df_ext = pd.read_csv("binary_class_result_with_train_mcc.csv")
fig, (ax_a, ax_b) = plt.subplots(1,2,figsize=(12,5),dpi=300)
x = np.arange(len(df_ext))
width = 0.4

ax_a.bar(x-width/2, df_ext["Test_AUC"], width, color="#2c7bb6")
ax_a.set_xticks(x)
ax_a.set_xticklabels(df_ext["Model"], rotation=45, ha="right")
ax_a.set_ylabel("Test‑AUC")
ax_a.set_title("Test‑set ROC‑AUC of different classification models")
ax_a.set_ylim(0.5,1.02)

ax_b.bar(x-width/2, df_ext["Test_MCC"], width, color="#d7191c")
ax_b.set_xticks(x)
ax_b.set_xticklabels(df_ext["Model"], rotation=45, ha="right")
ax_b.set_ylabel("Test‑MCC")
ax_b.set_title("Test‑set MCC of different classification models")
ax_b.set_ylim(-0.1, 1.02)

plt.tight_layout()
plt.savefig("model_comparison_bar.png",dpi=300)
plt.close()
print("[2] 多模型性能条形图保存 model_comparison_bar.png")

# ==============================================================
# ==========【新增模块3】AD适用域散点图（测试集max_tanimoto vs预测概率） ==========
test_df = pd.read_csv("testset_AD_analysis.csv")
plt.figure(figsize=(8,5),dpi=300)

mask_0 = test_df["y_true"] == 0
mask_1 = test_df["y_true"] == 1

plt.scatter(test_df.loc[mask_0,"max_tanimoto"], test_df.loc[mask_0,"y_prob_rf"], c="#1f77b4", label="True low‑activity (0)", s=60)
plt.scatter(test_df.loc[mask_1,"max_tanimoto"], test_df.loc[mask_1,"y_prob_rf"], c="#ff7f0e", label="True high‑activity (1)", s=60)

plt.axvline(x=0.35, linestyle="--", color="gray", label="AD threshold =0.35")

fp_mask = (test_df["y_true"]==0) & (test_df["y_pred_rf"]==1)
fn_mask = (test_df["y_true"]==1) & (test_df["y_pred_rf"]==0)
plt.scatter(test_df.loc[fp_mask,"max_tanimoto"], test_df.loc[fp_mask,"y_prob_rf"], marker="*", c="red", s=220, label="False Positive")
plt.scatter(test_df.loc[fn_mask,"max_tanimoto"], test_df.loc[fn_mask,"y_prob_rf"], marker="D", c="black", s=120, label="False Negative")

plt.xlabel("Max Tanimoto similarity to training set")
plt.ylabel("RF predicted probability (high‑activity)")
plt.legend()
plt.tight_layout()
plt.savefig("ad_scatter_testset.png",dpi=300)
plt.close()
print("[3] AD测试集散点图保存 ad_scatter_testset.png")

# ==============================================================
# ==========【新增模块4】物化特征相关性热力图（SI） ==========
X_num_df = df[feature_cols].copy()
corr_mat = X_num_df.corr(method="pearson")
plt.figure(figsize=(10,8),dpi=300)
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson correlation of physicochemical & DFT descriptors")
plt.tight_layout()
plt.savefig("physicochem_corr_heatmap.png",dpi=300)
plt.close()
print("[4] 物化特征相关性热力图保存 physicochem_corr_heatmap.png")

# ==============================================================
# ==========【新增模块5】多随机种子稳定性箱线图（SI） ==========
df_seed = pd.read_csv("multi_seed_stability.csv")
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,4),dpi=300)

data_cv = df_seed["cv_mean"].values
data_test = df_seed["test_auc"].values

ax1.boxplot(data_cv)
ax1.set_xticklabels(["CV‑AUC"])
ax1.set_ylim(0.6, 0.98)
ax1.set_title("CV‑AUC across different random seeds")

ax2.boxplot(data_test)
ax2.set_xticklabels(["Test‑AUC"])
ax2.set_ylim(0.6,0.98)
ax2.set_title("Test‑AUC across different random seeds")

plt.tight_layout()
plt.savefig("multi_seed_boxplot.png",dpi=300)
plt.close()
print("[5] 多随机种子稳定性箱线图保存 multi_seed_boxplot.png")

# ==============================================================
# ==========【模块6：SHAP分析】 ==========
# 使用调优后的best_rf进行SHAP分析
X_for_shap = X_train
full_feature_names = feature_cols + fp_names

# 创建SHAP解释器
explainer = shap.TreeExplainer(
    best_rf,
    feature_perturbation="interventional"
)
shap_values = explainer.shap_values(X_for_shap, check_additivity=False)

# 绘制SHAP蜂群图（仅物化特征）
shap.summary_plot(
    shap_values[:, :, 1][:, :len(feature_cols)],
    X_for_shap[:, :len(feature_cols)],
    feature_names=feature_cols,
    show=False,
    plot_size=(8, 5.5)
)
plt.tight_layout()
plt.savefig("shap_beeswarm_physicochem.png", dpi=300, bbox_inches="tight")
plt.close()

# 绘制SHAP条形图（所有特征）
shap.summary_plot(
    shap_values[:, :, 1],
    X_for_shap,
    feature_names=full_feature_names,
    plot_type="bar",
    show=False,
    plot_size=(8,4.5)
)
plt.tight_layout()
plt.savefig("shap_bar_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("[6] SHAP分析完成，图片已保存")
print("    - shap_beeswarm_physicochem.png (蜂群图，仅物化特征)")
print("    - shap_bar_importance.png (特征重要性条形图)")

# ==============================================================
# ==========【模块7】特征‑活性相关性分析 ==========
X_num_df = df[feature_cols].copy()
y_cont = df[target_reg_col].values   # 原始连续实验活性
y_bin = y_class.values              # 0/1二分类标签

res_corr = []
for feat in feature_cols:
    x_val = X_num_df[feat].values
    # 和连续活性的pearson
    corr_pearson = np.corrcoef(x_val, y_cont)[0,1]
    # 和二分类标签：点二列相关
    pb_res = pointbiserialr(x_val, y_bin)
    res_corr.append({
        "Feature": feat,
        "Pearson_vs_continuous_activity": round(corr_pearson,4),
        "Point‑biserial_vs_binary_label": round(pb_res.correlation,4),
        "p_value_point_biserial": round(pb_res.pvalue,4)
    })

df_feat_target_corr = pd.DataFrame(res_corr)
df_feat_target_corr.to_csv("feature_target_correlation.csv",index=False,encoding="utf-8-sig")

print("\n====特征‑活性相关性结果 ====")
print(df_feat_target_corr.to_string(index=False))

# 绘图：点二列相关条形图（SI图）
fig = plt.figure(figsize=(10,6),dpi=300)
feat_names = df_feat_target_corr["Feature"].values
pb_corr = df_feat_target_corr["Point‑biserial_vs_binary_label"].values
colors = ["#d62728" if x<0 else "#1f77b4" for x in pb_corr]
plt.barh(feat_names, pb_corr, color=colors)
plt.axvline(x=0,color="black",linestyle="--")
plt.xlabel("Point‑biserial correlation (vs binary high/low activity)")
plt.title("Correlation between physicochemical descriptors and re‑activation activity label")
plt.tight_layout()
plt.savefig("feature_target_pointbiserial_bar.png",dpi=300)
plt.close()
print("特征‑活性点二列相关条形图保存 feature_target_pointbiserial_bar.png")

# ==============================================================
# ==========保存全部模型 ==========
os.makedirs("saved_all_models", exist_ok=True)
for m_name, est in models.items():
    joblib.dump(est, f"saved_all_models/{m_name}.pkl")
print("全部7个基础模型已保存至 saved_all_models/")

print("\n====全部训练和绘图完成====")