import numpy as np, pandas as pd, joblib, json
from pathlib import Path

ART = Path(r"artifacts/synth")
xgb = joblib.load(ART / "xgb_binary.joblib")
cfg = json.load(open(ART / "config.json"))

FEATURES  = cfg["feature_order"]
CAT_FEAT  = cfg["cat_feat"]
CAT_MAP   = cfg["cat_mapping"]
THRESH    = cfg["threshold"]
MAX_DAYS  = cfg["max_days"]

def _prob_to_days(p, max_days=MAX_DAYS):
    d = 1 + int(round((1.0 - float(p)) * (max_days - 1)))
    return int(max(1, min(max_days, d)))

def predict_and_schedule(df_rows: pd.DataFrame) -> pd.DataFrame:
    X = df_rows[FEATURES].copy()
    # Apply in-memory mapping if cat feature is not numeric
    if CAT_MAP is not None and not np.issubdtype(X[CAT_FEAT].dtype, np.number):
        X[CAT_FEAT] = X[CAT_FEAT].map(CAT_MAP).fillna(-1).astype(int)
    proba = xgb.predict_proba(X.values)[:, 1]
    out = df_rows[["unit_id"]].copy() if "unit_id" in df_rows.columns else pd.DataFrame(index=df_rows.index)
    out["proba_failure"] = proba
    out["needs_maintenance"] = (proba >= THRESH).astype(int)
    out["days_to_maintenance"] = [_prob_to_days(p) for p in proba]
    out["message"] = np.where(
        out["needs_maintenance"]==1,
        out.apply(lambda r: f"in {int(r['days_to_maintenance'])} days machine {r.get('unit_id','UNKNOWN')} needs maintenance", axis=1),
        ""
    )
    return out
