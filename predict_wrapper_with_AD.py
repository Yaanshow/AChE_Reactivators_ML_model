import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import warnings

# 屏蔽Morgan弃用警告
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
    特征：10个物化描述符 + Morgan2 1024指纹
    """
    def __init__(self, model_path: str, scaler_path: str, ad_fps_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.train_fps = joblib.load(ad_fps_path)
        self.ad_tool = ApplicabilityDomainTanimoto(self.train_fps, threshold=0.35)

        self.morgan_radius = 2
        self.morgan_n_bits = 1024

        # !!! 和训练集严格保持一致，不要多余空格
        self.feature_cols = [
            "HOMO-1",
            "LUMO_HOMO_gap (eV)",
            "Total_Score",
            "Crash",
            "Polar",
            "Molar Volume",
            "O Natural Charge",
            "N Natural Charge",
            "Tot(D)",
            "pKa"
        ]
        self.smi_col = "smi"

    def _get_fp(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            warnings.warn(f"无效SMILES: {smi}")
            return np.zeros(self.morgan_n_bits, dtype=np.float64), None
        fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.morgan_radius, nBits=self.morgan_n_bits)
        return np.array(fp_obj, dtype=np.float64), fp_obj

    def featurize(self, df_in):
        desc_matrix = df_in[self.feature_cols].copy().values.astype(np.float64)
        fp_arr_list = []
        fp_obj_list = []
        for s in df_in[self.smi_col]:
            arr, obj = self._get_fp(s)
            fp_arr_list.append(arr)
            fp_obj_list.append(obj)
        fp_matrix = np.stack(fp_arr_list, axis=0)
        X_raw = np.hstack([desc_matrix, fp_matrix])
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
    # ========== 正式论文使用 rf_tuned_best.pkl ==========
    wrapper = OximReactivationModel(
        model_path="rf_tuned_best.pkl",
        scaler_path="standard_scaler.pkl",
        ad_fps_path="ad_train_fps.joblib"
    )

    df_input = pd.read_excel("WZ_redesigned_compounds.xlsx")
    result = wrapper.predict(df_input)

    # 排序：优先AD内，再按预测概率降序
    result_sort = result.sort_values(["inside_AD", "pred_high_activity_prob"], ascending=[False, False]).reset_index(drop=True)

    out_csv = "virtual_screening_result_with_AD.csv"
    result_sort.to_csv(out_csv, index=False, encoding="utf‑8‑sig")
    print(f"已输出结果至 {out_csv}")
    print("\n==== Top10候选 (smi | prob | max_tanimoto | inside_AD) ====")
    show_cols = ["smi", "pred_high_activity_prob", "max_tanimoto", "inside_AD"]
    print(result_sort[show_cols].head(10).to_string(index=False))

    # 简单统计
    total = len(result_sort)
    ad_in = sum(result_sort["inside_AD"])
    ad_out = total - ad_in
    print(f"\n数据集总分子:{total}, AD域内分子:{ad_in}, AD域外(外推):{ad_out}")
