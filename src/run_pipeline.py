"""ZİNCİR-TR ana boru hattı: çek -> eşitle -> fizik -> hibrit -> değerlendir -> dışa aktar."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_and_climate as E  # noqa: E402
import fetch_and_harmonize as F  # noqa: E402
import hybrid_and_feedback as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = E.MODELS


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else round(float(o), 3)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%dT%H:%MZ")
    return o


def series_list(s: pd.Series, nd: int = 2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def iso(idx):
    return [t.strftime("%Y-%m-%dT%H:%MZ") for t in idx]


def update_ledger(frames: dict, cfg: dict, now: pd.Timestamp, log):
    keep = cfg["run"]["ledger_keep_days"]
    cols_models = MODELS + ["M5_Hibrit_GFS_YZ_p10", "M5_Hibrit_GFS_YZ_p90", "M6_Hibrit_AIFS_YZ_p10", "M6_Hibrit_AIFS_YZ_p90"]
    new = []
    for sid, f in frames.items():
        w = f[f.index >= now - pd.Timedelta(days=cfg["run"]["eval_days"])]
        d = w[["ref"] + cols_models].copy()
        for c in ("v_meso", "temp", "aod"):
            d[c] = w[c]
        d["site"] = sid
        d["issue_date"] = now.strftime("%Y-%m-%d")
        d["kind"] = np.where(w.index >= now, "tahmin", "geçmiş_karne")
        d["time"] = w.index
        new.append(d.reset_index(drop=True))
    new = pd.concat(new, ignore_index=True)
    path = ROOT / "data" / "master_ledger.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        led = pd.concat([old, new], ignore_index=True)
    else:
        led = new
    led = led.drop_duplicates(["site", "time", "issue_date"], keep="last")
    # eski satırların eksik referansını yeni ERA5 değerleriyle tamamla
    newref = new.dropna(subset=["ref"]).drop_duplicates(["site", "time"], keep="last").set_index(["site", "time"])["ref"]
    key = pd.MultiIndex.from_frame(led[["site", "time"]])
    fill = pd.Series(newref.reindex(key).to_numpy(), index=led.index)
    led["ref"] = led["ref"].fillna(fill)
    led = led[led["time"] >= now - pd.Timedelta(days=keep)].sort_values(["site", "issue_date", "time"])
    led.to_parquet(path, index=False)
    led.to_csv(ROOT / "data" / "master_ledger.csv", index=False, encoding="utf-8-sig", float_format="%.3f")
    log(f"Ana kütük: {len(led)} satır -> data/master_ledger.parquet / .csv")


def site_payload(sid, site, f, rows, cat, rad, clm, diag, data, clim, now, cfg):
    cap = site["capacity_mw"]
    fut = f[f.index >= now]
    past = f[(f.index >= now - pd.Timedelta(days=cfg["run"]["eval_days"])) & (f.index < now)]
    met_cols = ["v10", "vhub", "gust", "v_gfs", "v_aifs", "alpha_meso"] if site["type"] == "wind" else ["ghi", "dni", "kt", "v10"]
    met_cols += ["cloud", "aod", "temp"]

    def block(d):
        out = {"time": iso(d.index)}
        for m in MODELS:
            out[m] = series_list(d[m])
        for m in ("M5_Hibrit_GFS_YZ", "M6_Hibrit_AIFS_YZ"):
            out[m + "_p10"] = series_list(d[m + "_p10"])
            out[m + "_p90"] = series_list(d[m + "_p90"])
        for c in met_cols:
            if c in d:
                out[c] = series_list(d[c], 3)
        return out

    fb = block(fut)
    gb = block(past)
    gb["ref"] = series_list(past["ref"])
    # canlı durum (şu anki saat)
    cur = f.loc[now] if now in f.index else fut.iloc[0]
    live = {"time": now.strftime("%Y-%m-%dT%H:%MZ"), "temp": cur["temp"], "aod": cur["aod"], "cloud": cur["cloud"],
            "M6": cur["M6_Hibrit_AIFS_YZ"], "M6_p10": cur["M6_Hibrit_AIFS_YZ_p10"], "M6_p90": cur["M6_Hibrit_AIFS_YZ_p90"],
            "M5": cur["M5_Hibrit_GFS_YZ"], "cf_pct": (cur["M6_Hibrit_AIFS_YZ"] / cap * 100) if pd.notna(cur["M6_Hibrit_AIFS_YZ"]) else None}
    if site["type"] == "wind":
        live.update({"vhub": cur["vhub"], "gust": cur["gust"], "v10": cur["v10"], "alpha": cur["alpha_meso"]})
    else:
        live.update({"ghi": cur["ghi"], "dni": cur["dni"], "v10": cur["v10"]})
    events = []
    if not cat.empty:
        c = cat[(cat["end"] >= now) | (cat["period"] == "geçmiş")]
        events = [{"model": r.model, "type": r.type, "start": r.start, "end": r.end, "hours": r.hours,
                   "peak": r.peak, "unit": r.unit, "period": r.period, "detail": r.detail} for r in c.itertuples()]
    return {"id": sid, "label": site["label"], "type": site["type"], "capacity_mw": cap, "lat": site["lat"],
            "lon": site["lon"], "hub_m": site.get("hub_m"), "live": live, "radar": rad, "forecast": fb, "ghost": gb,
            "leaderboard": rows, "climate": clm, "events": events, "diagnostics": diag,
            "model_names": data["model_names"], "aod_median": data["aod_median"],
            "last_ref_time": diag["last_label"]}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
    now = pd.Timestamp.now(tz="UTC").floor("h")
    log = lambda *a: print(*a, flush=True)
    t0 = time.time()
    log(f"ZİNCİR-TR çalışıyor | şimdi (UTC) = {now}")
    sites = args.sites or list(cfg["sites"])
    payloads, frames, cats = {}, {}, []
    for sid in sites:
        site = cfg["sites"][sid]
        log(f"=== {sid} ===")
        data = F.load_site(sid, cfg, now, log)
        clim = H.climatology(site, cfg, data)
        f = H.build_frame(sid, site, data, cfg, clim)
        f, diag = H.run_hybrid(site, f, cfg, now, log)
        rows = E.score(site, f, cfg, now)
        cat = E.detect_events(sid, site, f, cfg, clim, now)
        rad = E.radar(cat, now)
        clm = E.climate(site, cfg, data, clim, f, rows, now)
        for r in rows:
            log(f"  {r['model']:<22} RMSE={r['rmse']:.2f} nMAE={r['nmae']:.1f}% bias={r['bias']:.2f} "
                f"RampDA={r['ramp_da'] if r['ramp_da'] is None else round(r['ramp_da'], 1)} cov={r['coverage']}")
        payloads[sid] = site_payload(sid, site, f, rows, cat, rad, clm, diag, data, clim, now, cfg)
        frames[sid] = f
        if not cat.empty:
            cats.append(cat)
    update_ledger(frames, cfg, now, log)
    allcat = pd.concat(cats, ignore_index=True) if cats else pd.DataFrame()
    allcat.to_csv(ROOT / "data" / "anomaly_catalog.csv", index=False, encoding="utf-8-sig")
    log(f"Anomali kataloğu: {len(allcat)} olay")
    meta = {"generated_at": now.strftime("%Y-%m-%dT%H:%MZ"), "run_seconds": round(time.time() - t0, 1),
            "reference_note": "Referans güç = ERA5 reanalizinden aynı fizik zinciriyle hesaplanan güç (SCADA değildir).",
            "aod_note": "CAMS ufku (~4-5 gün) ötesinde AOD/toz, gerçek CAMS kaydının medyanıyla sabitlenir.",
            "site_order": sites}
    out = clean({"meta": meta, "sites": payloads})
    web = ROOT / "docs" / "web_data"
    web.mkdir(parents=True, exist_ok=True)
    js = json.dumps(out, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (web / "observability_data.json").write_text(js, encoding="utf-8")
    (web / "data_bundle.js").write_text("window.ZINCIR_DATA = " + js + ";\n", encoding="utf-8")
    log(f"Web verisi yazıldı ({len(js) / 1e6:.2f} MB). Süre: {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()
