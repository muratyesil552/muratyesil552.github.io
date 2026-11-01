"""Faktör D: M1-M6 üretimi, Çok Ölçekli Kalıntı İstifleme (LightGBM), ACI konformal bantlar,
G2 (kalıntı + ACI), G5 (iklimsel hat -> anlık hat) ve G4 girdileri."""
from __future__ import annotations

import numpy as np
import pandas as pd

import physics_chain as P

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


# --------------------------------------------------------------------------- G5: iklim hattı
def climatology(site: dict, cfg: dict, data: dict) -> dict:
    """ERA5 uzun kaydından persentil eşikleri ve (ay, 3 saatlik dilim) güç dağılımı."""
    era = P.fill_aero(data["era5"], data["aod_median"], data["dust_median"])
    era = era[era["v10"].notna()]
    pref = P.site_power(era, site, cfg, "full", data["aod_median"])
    cap = site["capacity_mw"]
    clim = {"era": era, "pref": pref}
    local = era.index + pd.Timedelta(hours=site["utc_offset"])
    tab = {}
    for (m, b), g in pref.groupby([local.month, local.hour // 3]):
        tab[(int(m), int(b))] = np.sort(g.dropna().to_numpy())
    clim["table"] = tab
    ramp = (pref.shift(-3) - pref).abs() / cap
    clim["ramp_p90"] = float(ramp.quantile(0.90)) * cap
    clim["ramp_p95"] = float(ramp.quantile(0.95)) * cap
    if site["type"] == "wind":
        clim["v_p90"] = float(era["vhub"].quantile(0.90))
        clim["v_p95"] = float(era["vhub"].quantile(0.95))
        night = ((local.hour >= 20) | (local.hour < 6))
        clim["alpha_night_p90"] = float(era.loc[night, "alpha"].quantile(0.90))
    else:
        clim["v_p90"] = clim["v_p95"] = np.nan
        clim["alpha_night_p90"] = np.nan
    clim["aod_p95"] = data["aod_p95"]
    return clim


def clim_percentile(power: pd.Series, clim: dict, site: dict) -> pd.Series:
    local = power.index + pd.Timedelta(hours=site["utc_offset"])
    out = np.full(len(power), np.nan)
    keys = list(zip(local.month, local.hour // 3))
    vals = power.to_numpy(float)
    for i, (k, v) in enumerate(zip(keys, vals)):
        arr = clim["table"].get((int(k[0]), int(k[1])))
        if arr is not None and len(arr) and not np.isnan(v):
            out[i] = np.searchsorted(arr, v, side="right") / len(arr)
    return pd.Series(out, index=power.index)


# --------------------------------------------------------------------------- çerçeve (M1-M4 + özellikler)
def build_frame(site_id: str, site: dict, data: dict, cfg: dict, clim: dict) -> pd.DataFrame:
    am, dm = data["aod_median"], data["dust_median"]
    md = {k: P.fill_aero(v, am, dm) for k, v in data["models"].items()}
    meso = P.blend_meso(md["IFS"], md["ICON"], site)
    meso = P.fill_aero(meso.drop(columns=["aero_filled"], errors="ignore"), am, dm)
    # aod/dust meso için IFS'ten gelir (CAMS aynı saha için ortaktır)
    gfs, aifs = md["GFS"], md["AIFS"]
    idx = meso.index.union(gfs.index).union(aifs.index)
    f = pd.DataFrame(index=idx)
    f["M1_GFS_Temel"] = P.site_power(gfs, site, cfg, "raw", am)
    f["M2_AIFS_Temel"] = P.site_power(aifs, site, cfg, "raw", am)
    f["M3_OrtaOlcek_Fizik"] = P.site_power(meso, site, cfg, "raw", am)
    f["M4_OrtaOlcek_Mikro"] = P.site_power(meso, site, cfg, "full", am)
    f["pg_full"] = P.site_power(gfs, site, cfg, "full", am)
    f["pa_full"] = P.site_power(aifs, site, cfg, "full", am)
    # Referans: ERA5 gücü (SCADA değildir)
    f["ref"] = clim["pref"].reindex(idx)
    # meteoroloji (görüntüleme + özellik)
    if site["type"] == "wind":
        key = "vhub"
        f["v_gfs"], f["v_aifs"], f["v_meso"] = gfs["vhub"], aifs["vhub"], meso["vhub"]
        f["alpha_meso"], f["alpha_aifs"] = meso["alpha"], aifs["alpha"]
        f["gust"], f["dir"] = meso["gust"], meso["dir"]
        f["v10"], f["vhub"] = meso["v10"], meso["vhub"]
    else:
        f["v_gfs"], f["v_aifs"], f["v_meso"] = gfs["ghi"], aifs["ghi"], meso["ghi"]
        f["ghi"], f["dni"], f["kt"] = meso["ghi"], meso["dni"], meso["kt"]
        f["v10"] = meso["v10"]
    f["temp"], f["cloud"], f["aod"] = meso["temp"], meso["cloud"], meso["aod"]
    f["pres"] = meso["pres"]
    f["zenith"] = meso["zenith"]
    f["ifs_icon_blend"] = meso["icon_used"]
    # özellikler: gradyanlar (nedensel)
    for k in (1, 3):
        f[f"dv_dt{k}"] = f["v_meso"] - f["v_meso"].shift(k)
        f[f"dp_dt{k}"] = f["pres"] - f["pres"].shift(k)
        f[f"dghi_dt{k}"] = (meso["ghi"] - meso["ghi"].shift(k)) if "ghi" in meso else np.nan
    f["d_large"] = f["v_aifs"] - f["v_gfs"]
    local = f.index + pd.Timedelta(hours=site["utc_offset"])
    f["hr_sin"], f["hr_cos"] = np.sin(2 * np.pi * local.hour / 24), np.cos(2 * np.pi * local.hour / 24)
    doy = local.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["clim_pct"] = clim_percentile(f["M4_OrtaOlcek_Mikro"], clim, site)
    # model bazlı olay girdileri (anomali motoru)
    if site['type'] == 'wind':
        f['alpha_gfs'] = gfs['alpha']
        f['storm_gfs'] = P.storm_mask(gfs['vhub'].to_numpy(float), gfs['gust'].to_numpy(float), cfg['thresholds'])
        f['storm_aifs'] = P.storm_mask(aifs['vhub'].to_numpy(float), None, cfg['thresholds'])
    else:
        f['kt_gfs'], f['kt_aifs'], f['kt_meso'] = gfs['kt'], aifs['kt'], meso['kt']
    f['aod_raw_filled'] = meso['aero_filled']
    # fırtına/gece maskeleri (hibrit çıktısına fiziksel kısıt)
    if site["type"] == "wind":
        f["force0"] = P.storm_mask(meso["vhub"].to_numpy(float), meso["gust"].to_numpy(float), cfg["thresholds"])
    else:
        f["force0"] = (meso["ghi"].fillna(1.0) <= 1.0)
    return f


COMMON = ["M3_OrtaOlcek_Fizik", "M4_OrtaOlcek_Mikro", "v_meso", "dv_dt1", "dv_dt3", "dp_dt1", "dp_dt3",
          "dghi_dt1", "dghi_dt3", "alpha_meso", "aod", "hr_sin", "hr_cos", "doy_sin", "doy_cos", "clim_pct"]
FEATS = {
    "M5_Hibrit_GFS_YZ": COMMON + ["M1_GFS_Temel", "pg_full", "v_gfs"],
    "M6_Hibrit_AIFS_YZ": COMMON + ["M2_AIFS_Temel", "pa_full", "v_aifs", "d_large", "alpha_aifs"],
}
DRIVER = {"M5_Hibrit_GFS_YZ": "pg_full", "M6_Hibrit_AIFS_YZ": "pa_full"}


def _make_model(cfg):
    p = cfg["hybrid"]["lgbm"]
    if LGBMRegressor is not None:
        return LGBMRegressor(objective="regression", n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                             num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                             subsample=p["subsample"], subsample_freq=p["subsample_freq"],
                             colsample_bytree=p["colsample_bytree"], random_state=7, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=p["n_estimators"], learning_rate=p["learning_rate"], random_state=7)


def _ramp_weights(f: pd.DataFrame, cap: float, cfg: dict) -> np.ndarray:
    """Rampa anlarında ve yön hatası yapılan anlarda daha yüksek ağırlık."""
    h = cfg["hybrid"]
    d_ref = f["ref"].shift(-3) - f["ref"]
    d_phys = f["M4_OrtaOlcek_Mikro"].shift(-3) - f["M4_OrtaOlcek_Mikro"]
    mag = np.minimum(1.0, d_ref.abs() / (cfg["thresholds"]["ramp_frac_3h"] * cap))
    wrong = (np.sign(d_ref) != np.sign(d_phys)) & (d_ref.abs() >= cfg["thresholds"]["ramp_da_min_frac"] * cap)
    w = 1.0 + h["ramp_weight"] * mag + h["ramp_weight"] * wrong.astype(float)
    return w.fillna(1.0).to_numpy()


# --------------------------------------------------------------------------- ACI
def aci_walk(y, pred, buf0, alpha_target, gamma, window):
    """alpha_t = alpha_{t-1} + gamma * (alpha_target - err_t); bant yarı genişliği = tampon |hata| (1 - alpha_t) kuantili."""
    buf = list(buf0[-window:])
    a = alpha_target
    lo, hi, alphas = [], [], []
    for yt, pt in zip(y, pred):
        if np.isnan(pt) or not buf:
            lo.append(np.nan); hi.append(np.nan); alphas.append(a); continue
        q = np.max(buf) * 1.5 if a <= 0 else np.quantile(buf, min(1.0, max(0.0, 1 - a)))
        lo.append(pt - q); hi.append(pt + q); alphas.append(a)
        if not np.isnan(yt):
            err = float(yt < pt - q or yt > pt + q)
            a = a + gamma * (alpha_target - err)
            a = min(0.999, max(0.0, a))
            buf.append(abs(yt - pt))
            buf = buf[-window:]
    return np.array(lo), np.array(hi), np.array(alphas), a, buf


def _fit(f: pd.DataFrame, name: str, cutoff: pd.Timestamp, cap: float, cfg: dict):
    feats = [c for c in FEATS[name] if c in f and f[c].notna().any()]
    tr = f[(f.index < cutoff) & f["ref"].notna() & f["M4_OrtaOlcek_Mikro"].notna() & f[DRIVER[name]].notna()]
    y = (tr["ref"] - tr["M4_OrtaOlcek_Mikro"]).to_numpy()
    w = _ramp_weights(f, cap, cfg)[f.index.get_indexer(tr.index)]
    m = _make_model(cfg)
    m.fit(tr[feats], y, sample_weight=w)
    return m, feats, len(tr)


def _predict(m, feats, f, name, cap):
    ok = f["M4_OrtaOlcek_Mikro"].notna() & f[DRIVER[name]].notna()
    pred = pd.Series(np.nan, index=f.index)
    e = m.predict(f.loc[ok, feats])
    p = np.clip(f.loc[ok, "M4_OrtaOlcek_Mikro"].to_numpy() + e, 0, cap)
    p = np.where(f.loc[ok, "force0"].to_numpy(bool), 0.0, p)
    pred[ok] = p
    return pred


def run_hybrid(site: dict, f: pd.DataFrame, cfg: dict, now: pd.Timestamp, log=print) -> tuple[pd.DataFrame, dict]:
    """M5/M6: değerlendirme katı (eğitim < eval_start-21g, kalibrasyon 21g, ACI yürüyüşü eval penceresi)
    ve canlı katı (son 14 günlük etiket hariç eğitim)."""
    cap = site["capacity_mw"]
    h = cfg["hybrid"]
    eval_start = now - pd.Timedelta(days=cfg["run"]["eval_days"])
    cal_start = eval_start - pd.Timedelta(days=21)
    last_label = f.loc[f["ref"].notna() & (f.index < now), "ref"].index.max()
    live_cut = last_label - pd.Timedelta(days=14)
    diag = {"last_label": last_label.isoformat(), "eval_start": eval_start.isoformat()}
    for name in FEATS:
        # ---- değerlendirme katı
        m, feats, n = _fit(f, name, cal_start, cap, cfg)
        pred = _predict(m, feats, f, name, cap)
        cal = f[(f.index >= cal_start) & (f.index < eval_start) & f["ref"].notna()]
        sc = (cal["ref"] - pred.reindex(cal.index)).abs().dropna().to_numpy()
        win = f[(f.index >= eval_start) & (f.index < now)]
        lo, hi, _, a_end, buf = aci_walk(win["ref"].to_numpy(), pred.reindex(win.index).to_numpy(), sc,
                                         h["alpha_target"], h["gamma_aci"], h["aci_window"])
        res = pd.DataFrame({"lo": lo, "hi": hi}, index=win.index)
        out = pd.Series(np.nan, index=f.index)
        out_lo, out_hi = out.copy(), out.copy()
        out[win.index] = pred.reindex(win.index)
        out_lo[win.index], out_hi[win.index] = res["lo"], res["hi"]
        # ---- canlı kat
        m2, feats2, n2 = _fit(f, name, live_cut, cap, cfg)
        pred2 = _predict(m2, feats2, f, name, cap)
        q = np.max(buf) * 1.5 if a_end <= 0 else np.quantile(buf, min(1.0, max(0.0, 1 - a_end)))
        fut = f.index >= now
        out[fut] = pred2[fut]
        out_lo[fut], out_hi[fut] = pred2[fut] - q, pred2[fut] + q
        f[name] = out
        f[name + "_p10"] = out_lo.clip(lower=0)
        f[name + "_p90"] = out_hi.clip(upper=cap)
        imp = pd.Series(m2.feature_importances_, index=feats2).sort_values(ascending=False) if hasattr(m2, "feature_importances_") else pd.Series(dtype=float)
        diag[name] = {"n_train_eval": int(n), "n_train_live": int(n2), "aci_alpha_final": float(a_end),
                      "band_halfwidth_mw": float(q), "top_features": {k: int(v) for k, v in imp.head(6).items()}}
        log(f"  {name}: eğitim={n}/{n2}, ACI alpha_son={a_end:.3f}, yarı-bant={q:.2f} MW")
    f["is_future"] = f.index >= now
    return f, diag
