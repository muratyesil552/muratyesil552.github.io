"""Doğrulama karnesi, anomali/kesinti motoru (G5 eşikleri) ve iklimsel verim P50/P75/P90 (G4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import physics_chain as P

MODELS = ["M1_GFS_Temel", "M2_AIFS_Temel", "M3_OrtaOlcek_Fizik", "M4_OrtaOlcek_Mikro",
          "M5_Hibrit_GFS_YZ", "M6_Hibrit_AIFS_YZ"]
DRIVER_OF = {"M1_GFS_Temel": "gfs", "M2_AIFS_Temel": "aifs"}  # diğerleri orta ölçek


# --------------------------------------------------------------------------- metrikler
def _events(p: pd.Series, thr: float) -> pd.Series:
    return ((p.shift(-3) - p).abs() >= thr).fillna(False)


def _dilate(b: pd.Series, k: int) -> pd.Series:
    return b.astype(int).rolling(2 * k + 1, center=True, min_periods=1).max().astype(bool)


def score(site: dict, f: pd.DataFrame, cfg: dict, now: pd.Timestamp) -> list[dict]:
    cap = site["capacity_mw"]
    th = cfg["thresholds"]
    start = now - pd.Timedelta(days=cfg["run"]["eval_days"])
    w = f[(f.index >= start) & (f.index < now) & f["ref"].notna()]
    ref = w["ref"]
    d_ref = ref.shift(-3) - ref
    ev_ref = _events(ref, th["ramp_frac_3h"] * cap)
    rows = []
    for m in MODELS:
        p = w[m]
        ok = p.notna()
        if ok.sum() < 24:
            continue
        e = (p - ref)[ok]
        d_m = p.shift(-3) - p
        ramp_mask = (d_ref.abs() >= th["ramp_da_min_frac"] * cap) & d_m.notna()
        ra = float((np.sign(d_m[ramp_mask]) == np.sign(d_ref[ramp_mask])).mean() * 100) if ramp_mask.sum() > 5 else None
        ev_m = _events(p, th["ramp_frac_3h"] * cap)
        k = th["event_tol_h"]
        hits = (ev_ref & _dilate(ev_m, k)).sum()
        n_ref = ev_ref.sum()
        n_mod = ev_m.sum()
        fa = (ev_m & ~_dilate(ev_ref, k)).sum()
        miss = n_ref - hits
        pod = float(hits / n_ref) if n_ref > 0 else None
        far = float(fa / n_mod) if n_mod > 0 else None
        csi = float(hits / (hits + miss + fa)) if (hits + miss + fa) > 0 else None
        cov = None
        if m in ("M5_Hibrit_GFS_YZ", "M6_Hibrit_AIFS_YZ"):
            lo, hi = w[m + "_p10"], w[m + "_p90"]
            okb = lo.notna() & hi.notna() & p.notna()
            if okb.sum() > 0:
                cov = float(((ref[okb] >= lo[okb]) & (ref[okb] <= hi[okb])).mean() * 100)
        rows.append({"model": m, "n": int(ok.sum()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "nmae": float(e.abs().mean() / cap * 100), "bias": float(e.mean()),
                     "ramp_da": ra, "pod": pod, "far": far, "csi": csi, "coverage": cov,
                     "n_ref_events": int(n_ref)})
    return rows


# --------------------------------------------------------------------------- anomali motoru
def _episodes(flag: pd.Series, gap: int = 2):
    idx = np.flatnonzero(flag.to_numpy(bool))
    if len(idx) == 0:
        return []
    eps, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            eps.append((s, prev)); s = i
        prev = i
    eps.append((s, prev))
    return eps


def detect_events(site_id: str, site: dict, f: pd.DataFrame, cfg: dict, clim: dict, now: pd.Timestamp) -> pd.DataFrame:
    cap, th = site["capacity_mw"], cfg["thresholds"]
    win = f[(f.index >= now - pd.Timedelta(days=cfg["run"]["eval_days"]))]
    local_h = (win.index + pd.Timedelta(hours=site["utc_offset"])).hour
    night = pd.Series((local_h >= 20) | (local_h < 6), index=win.index)
    recs = []

    def add(model, typ, flag, series=None, unit="", sign=None, desc=""):
        for s, e in _episodes(flag):
            seg = win.index[s:e + 1]
            val = None
            if series is not None:
                seg_v = series.iloc[s:e + 1]
                val = float(seg_v.iloc[np.nanargmax(np.abs(seg_v.to_numpy()))]) if seg_v.notna().any() else None
            recs.append({"site": site_id, "model": model, "type": typ, "start": seg[0], "end": seg[-1],
                         "hours": int(len(seg)), "peak": None if val is None else round(val, 2), "unit": unit,
                         "period": "gelecek" if seg[-1] >= now else "geçmiş", "detail": desc})

    for m in MODELS + ["ERA5_Referans"]:
        col = "ref" if m == "ERA5_Referans" else m
        p = win[col]
        if p.notna().sum() < 6:
            continue
        d3 = p.shift(-3) - p
        down = (d3 <= -th["ramp_frac_3h"] * cap).fillna(False)
        up = (d3 >= th["ramp_frac_3h"] * cap).fillna(False)
        add(m, "Ani Rampa Düşüşü", down, d3, "MW/3sa", desc=f"≥%{int(th['ramp_frac_3h']*100)} kurulu güç / 3 saat")
        add(m, "Ani Rampa Artışı", up, d3, "MW/3sa", desc=f"≥%{int(th['ramp_frac_3h']*100)} kurulu güç / 3 saat")
        if m == "ERA5_Referans":
            continue
        drv = DRIVER_OF.get(m, "meso")
        if site["type"] == "wind":
            storm = win[f"storm_{drv}"] if drv != "meso" else win["force0"]
            v = win[f"v_{drv}"]
            add(m, "Fırtına Kesintisi (Cut-Out)", storm.astype(bool), v, "m/s",
                desc=f"hub ≥ {th['cutout_speed']} m/s" + (f" veya hamle ≥ {th['cutout_gust']} m/s" if drv != "aifs" else ""))
            hi = (v >= clim["v_p95"]) & ~storm.astype(bool)
            add(m, "İklimsel Aşırı Rüzgâr (≥P95)", hi.fillna(False), v, "m/s", desc=f"ERA5 P95 = {clim['v_p95']:.1f} m/s (G5)")
            al = win["alpha_gfs"] if drv == "gfs" else (win["alpha_aifs"] if drv == "aifs" else win["alpha_meso"])
            lj = night & (al >= max(th["lljs_alpha_min"], clim["alpha_night_p90"])) & (v >= 6)
            add(m, "Gece Düşük Seviyeli Jet (LLJ)", lj.fillna(False), al, "α",
                desc=f"gece α ≥ {max(th['lljs_alpha_min'], clim['alpha_night_p90']):.2f} (G5 P90)")
        else:
            kt = win[f"kt_{drv}"]
            day = (win["zenith"] < 80)
            dk = (kt.shift(-3) - kt)
            col_f = ((dk <= -th["cloud_collapse_dkt"]) & day & day.shift(-3, fill_value=False)).fillna(False)
            add(m, "Bulut Çöküşü", col_f, dk, "Δkt/3sa", desc=f"berraklık indeksi düşüşü ≥ {th['cloud_collapse_dkt']}")
            dust = (win["aod"] >= clim["aod_p95"]) & ~win["aod_raw_filled"]
            add(m, "Toz/Aerosol Olayı", dust.fillna(False), win["aod"], "AOD", desc=f"CAMS AOD ≥ P95 = {clim['aod_p95']:.2f}")
    cat = pd.DataFrame(recs)
    return cat


def radar(cat: pd.DataFrame, now: pd.Timestamp) -> dict:
    """+6 saat ve +24 saat acil uyarı listesi (ana model: M6)."""
    out = {}
    if cat.empty:
        return {"h6": [], "h24": []}
    c = cat[(cat["model"] == "M6_Hibrit_AIFS_YZ") & (cat["end"] >= now)]
    for key, hrs in (("h6", 6), ("h24", 24)):
        sel = c[c["start"] <= now + pd.Timedelta(hours=hrs)]
        out[key] = [{"type": r.type, "start": r.start.isoformat(), "end": r.end.isoformat(), "hours": r.hours,
                     "peak": r.peak, "unit": r.unit, "detail": r.detail} for r in sel.itertuples()]
    return out


# --------------------------------------------------------------------------- iklimsel verim (Hat 2) + G4
def _annual(s: pd.Series, years: list[int]) -> pd.Series:
    return s.groupby(s.index.year).sum().reindex(years)


def climate(site: dict, cfg: dict, data: dict, clim: dict, f: pd.DataFrame, rows: list[dict], now: pd.Timestamp) -> dict:
    cap = site["capacity_mw"]
    era = clim["era"]
    years = [y for y in sorted(set(era.index.year)) if y < now.year]
    # tam yıl koşulu
    cnt = era["v10"].groupby(era.index.year).count()
    years = [y for y in years if cnt.get(y, 0) >= 8000]
    am = data["aod_median"]
    raw = P.site_power(era, site, cfg, "raw", am)
    full = clim["pref"]
    aep_raw = _annual(raw, years)
    aep_full = _annual(full, years)

    def pstats(a):
        mu, sd = float(a.mean()), float(a.std(ddof=1))
        q1, q3, med = np.percentile(a, [25, 75, 50])
        return {"P50": mu, "P75": mu - 0.6745 * sd, "P90": mu - 1.2816 * sd, "sigma": sd,
                "rcov": float((q3 - q1) / med), "cf": float(mu / (cap * 8760) * 100)}
    classic = pstats(aep_raw)
    # G4: kısa vade hata varyansı -> uzun vade belirsizlik
    m6 = next((r for r in rows if r["model"] == "M6_Hibrit_AIFS_YZ"), None)
    g = pstats(aep_full)
    if m6:
        e6 = (f["M6_Hibrit_AIFS_YZ"] - f["ref"]).dropna()
        e6 = e6[(e6.index >= now - pd.Timedelta(days=cfg["run"]["eval_days"])) & (e6.index < now)]
        rho = float(e6.autocorr(1)) if len(e6) > 24 else 0.0
        rho = min(max(rho, 0.0), 0.95)
        s_rand = m6["rmse"] * np.sqrt(8760 * (1 + rho) / (1 - rho))
        s_bias = abs(m6["bias"]) * 8760
        sd_tot = float(np.sqrt(g["sigma"] ** 2 + s_rand ** 2 + s_bias ** 2))
        g.update({"sigma_inter": g["sigma"], "sigma_short_random": float(s_rand), "sigma_short_bias": float(s_bias),
                  "sigma": sd_tot, "P75": g["P50"] - 0.6745 * sd_tot, "P90": g["P50"] - 1.2816 * sd_tot,
                  "rho_lag1": rho, "rmse_m6": m6["rmse"], "bias_m6": m6["bias"]})
    # kayıp ağacı (yıllık ortalama MWh)
    sub = era[era.index.year.isin(years)]
    tree = P.wind_loss_tree(sub, site, cfg) if site["type"] == "wind" else P.solar_loss_tree(sub, site, cfg, am)
    n = max(len(years), 1)
    tree = {k: v / n for k, v in tree.items()}
    # G5 eşikleri
    g5 = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in clim.items()
          if k in ("v_p90", "v_p95", "ramp_p90", "ramp_p95", "alpha_night_p90", "aod_p95")}
    return {"years": years, "aep_raw": {int(y): float(v) for y, v in aep_raw.items()},
            "aep_full": {int(y): float(v) for y, v in aep_full.items()},
            "classic": classic, "g4": g, "loss_tree": tree, "g5_thresholds": g5, "capacity_mw": cap}
