import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import warnings

warnings.filterwarnings("ignore", message="DEPRECATION WARNING: please use MorganGenerator")
warnings.filterwarnings("ignore")


class ApplicabilityDomainTanimoto:
    def __init__(self, train_fps, threshold=0.35):
        self.train_fps = train_fps
        self.threshold = threshold

    def calc_max_tanimoto(self, mol_fp):
        if mol_fp is None:
            return np.nan
        sims = [TanimotoSimilarity(mol_fp, fp_tr) for fp_tr in self.train_fps]
        return max(sims)

    def check_domain(self, mol_fp):
        max_sim = self.calc_max_tanimoto(mol_fp)
        if np.isnan(max_sim):
            return max_sim, False
        return max_sim, bool(max_sim >= self.threshold)


class OximReactivationModel:
    """
    肟类VX‑hAChE重活化二分类预测封装
    特征：7个物化描述符 + Morgan2 1024指纹（已删除Total_Score、N Natural Charge和pKa）
    """
    def __init__(self, model_path: str, scaler_path: str, ad_fps_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.train_fps = joblib.load(ad_fps_path)
        self.ad_tool = ApplicabilityDomainTanimoto(self.train_fps, threshold=0.35)

        self.morgan_radius = 2
        self.morgan_n_bits = 1024

        # 与训练脚本严格一致（7个特征，已删除Total_Score、N Natural Charge和pKa）
        self.feature_cols = [
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
        self.smi_col = "smi"
        
        # 打印调试信息
        print(f"期望特征数（物化）: {len(self.feature_cols)}")
        print(f"期望特征数（指纹）: {self.morgan_n_bits}")
        print(f"期望特征总数: {len(self.feature_cols) + self.morgan_n_bits}")
        print(f"Scaler期望特征数: {self.scaler.n_features_in_}")

    def _get_fp(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            warnings.warn(f"无效SMILES: {smi}")
            return np.zeros(self.morgan_n_bits, dtype=np.float64), None
        fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.morgan_radius, nBits=self.morgan_n_bits)
        return np.array(fp_obj, dtype=np.float64), fp_obj

    def featurize(self, df_in):
        # 检查输入数据是否包含所有必需的特征列
        missing_cols = [col for col in self.feature_cols if col not in df_in.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少以下特征列: {missing_cols}")
        
        # 检查是否有已删除的列
        deleted_cols_check = ["Total_Score", "N Natural Charge", "pKa"]
        extra_cols = [col for col in df_in.columns if col in deleted_cols_check]
        if extra_cols:
            print(f"⚠️ 警告: 输入数据包含已删除的列 {extra_cols}，将被忽略")
        
        desc_matrix = df_in[self.feature_cols].copy().values.astype(np.float64)
        print(f"物化特征矩阵形状: {desc_matrix.shape}")
        
        fp_arr_list = []
        fp_obj_list = []
        for s in df_in[self.smi_col]:
            arr, obj = self._get_fp(s)
            fp_arr_list.append(arr)
            fp_obj_list.append(obj)
        fp_matrix = np.stack(fp_arr_list, axis=0)
        print(f"指纹矩阵形状: {fp_matrix.shape}")
        
        X_raw = np.hstack([desc_matrix, fp_matrix])
        print(f"原始特征矩阵形状: {X_raw.shape}")
        print(f"Scaler期望特征数: {self.scaler.n_features_in_}")
        
        # 检查特征数是否匹配
        if X_raw.shape[1] != self.scaler.n_features_in_:
            print(f"\n❌ 特征数不匹配！")
            print(f"   输入特征数: {X_raw.shape[1]}")
            print(f"   Scaler期望特征数: {self.scaler.n_features_in_}")
            
            # 尝试诊断问题
            if X_raw.shape[1] == 1031 and self.scaler.n_features_in_ == 1032:
                print(f"\n💡 诊断: 您的输入数据可能包含8个物化特征（含pKa）")
                print(f"   但新模型期望7个物化特征（已删除pKa）")
                print(f"   请确保输入数据已删除'pKa'列")
            elif X_raw.shape[1] == 1032 and self.scaler.n_features_in_ == 1031:
                print(f"\n💡 诊断: 您的输入数据可能缺少'pKa'列")
                print(f"   但使用的是旧模型（期望8个物化特征）")
                print(f"   请使用重新训练后的模型文件")
            elif X_raw.shape[1] == 1030 and self.scaler.n_features_in_ == 1031:
                print(f"\n💡 诊断: 您的输入数据可能缺少某个特征列")
                print(f"   请检查是否包含所有7个物化特征列")
            
            raise ValueError(
                f"特征数不匹配！输入特征数: {X_raw.shape[1]}, "
                f"Scaler期望特征数: {self.scaler.n_features_in_}"
            )
        
        X_scaled = self.scaler.transform(X_raw)
        return X_scaled, fp_obj_list

    def predict(self, df_in):
        X_scaled, fp_obj_list = self.featurize(df_in)
        proba_pos = self.model.predict_proba(X_scaled)[:, 1]
        pred_label = self.model.predict(X_scaled)

        df_out = df_in.copy()
        df_out["pred_high_activity_prob"] = np.round(proba_pos, 4)
        df_out["pred_class"] = pred_label

        max_tani_list = []
        in_ad_list = []
        for fpobj in fp_obj_list:
            mt, inad = self.ad_tool.check_domain(fpobj)
            max_tani_list.append(np.round(mt,4) if not np.isnan(mt) else mt)
            in_ad_list.append(inad)

        df_out["max_tanimoto"] = max_tani_list
        df_out["inside_AD"] = in_ad_list
        return df_out


if __name__ == "__main__":
    # 首先验证模型文件是否匹配
    print("=" * 60)
    print("检查模型文件...")
    print("=" * 60)
    
    try:
        scaler_check = joblib.load("standard_scaler.pkl")
        print(f"✅ standard_scaler.pkl")
        print(f"   期望特征数: {scaler_check.n_features_in_}")
        
        model_check = joblib.load("rf_tuned_best.pkl")
        print(f"✅ rf_tuned_best.pkl")
        print(f"   期望特征数: {model_check.n_features_in_}")
        
        # 判断模型版本
        if scaler_check.n_features_in_ == 1031:
            print(f"   ✅ 当前模型版本: 7个物化特征 (已删除Total_Score、N Natural Charge和pKa)")
            expected_features = 1031
        elif scaler_check.n_features_in_ == 1032:
            print(f"   ⚠️ 当前模型版本: 8个物化特征 (已删除Total_Score和N Natural Charge，仍包含pKa)")
            expected_features = 1032
        elif scaler_check.n_features_in_ == 1033:
            print(f"   ⚠️ 当前模型版本: 9个物化特征 (仅删除Total_Score，仍包含N Natural Charge和pKa)")
            expected_features = 1033
        elif scaler_check.n_features_in_ == 1034:
            print(f"   ⚠️ 当前模型版本: 10个物化特征 (包含Total_Score、N Natural Charge和pKa)")
            expected_features = 1034
        else:
            print(f"   ⚠️ 未知版本: {scaler_check.n_features_in_} 个特征")
            expected_features = scaler_check.n_features_in_
        
        if scaler_check.n_features_in_ != model_check.n_features_in_:
            print(f"⚠️ 警告: Scaler ({scaler_check.n_features_in_}) 和模型 ({model_check.n_features_in_}) 期望特征数不一致！")
        
        # 检查AD文件
        ad_check = joblib.load("ad_train_fps.joblib")
        print(f"✅ ad_train_fps.joblib")
        print(f"   AD训练指纹数量: {len(ad_check)}")
            
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
        print("   请确保以下文件存在:")
        print("   - standard_scaler.pkl")
        print("   - rf_tuned_best.pkl")
        print("   - ad_train_fps.joblib")
        exit(1)
    
    print("=" * 60)
    
    # 创建预测器
    wrapper = OximReactivationModel(
        model_path="rf_tuned_best.pkl",
        scaler_path="standard_scaler.pkl",
        ad_fps_path="ad_train_fps.joblib"
    )

    # 读取输入数据
    df_input = pd.read_excel("WZ_redesigned_compounds_7_features.xlsx")
    print(f"\n输入数据形状: {df_input.shape}")
    print(f"输入数据列名: {df_input.columns.tolist()}")
    
    # 检查输入数据是否包含已删除的特征
    deleted_cols = []
    if "Total_Score" in df_input.columns:
        deleted_cols.append("Total_Score")
    if "N Natural Charge" in df_input.columns:
        deleted_cols.append("N Natural Charge")
    if "pKa" in df_input.columns:
        deleted_cols.append("pKa")
    
    if deleted_cols:
        print(f"\n⚠️ 警告: 输入数据包含已删除的特征列: {deleted_cols}")
        print(f"   这些列将被忽略，不影响预测结果。")
        print(f"   建议从Excel文件中删除这些列以避免混淆。")
    
    # 检查是否缺少必需的特征
    required_cols = wrapper.feature_cols
    missing_required = [col for col in required_cols if col not in df_input.columns]
    if missing_required:
        print(f"\n❌ 错误: 输入数据缺少必需的特征列: {missing_required}")
        print(f"   请确保输入数据包含以下7个特征列:")
        for col in required_cols:
            print(f"   - {col}")
        exit(1)
    
    result = wrapper.predict(df_input)

    # 排序：优先AD内，再按预测概率降序
    result_sort = result.sort_values(["inside_AD", "pred_high_activity_prob"], ascending=[False, False]).reset_index(drop=True)

    out_csv = "virtual_screening_result_with_AD_7features.csv"
    result_sort.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已输出结果至 {out_csv}")
    print("\n==== Top10候选 (smi | prob | max_tanimoto | inside_AD) ====")
    show_cols = ["smi", "pred_high_activity_prob", "max_tanimoto", "inside_AD"]
    print(result_sort[show_cols].head(10).to_string(index=False))

    # 简单统计
    total = len(result_sort)
    ad_in = sum(result_sort["inside_AD"])
    ad_out = total - ad_in
    print(f"\n数据集总分子:{total}, AD域内分子:{ad_in}, AD域外(外推):{ad_out}")
    
    # 输出预测结果统计
    high_prob = sum(result_sort["pred_high_activity_prob"] >= 0.5)
    low_prob = total - high_prob
    print(f"预测高活性分子(≥0.5): {high_prob}, 预测低活性分子(<0.5): {low_prob}")