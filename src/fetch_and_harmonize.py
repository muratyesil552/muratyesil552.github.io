"""Faktör A/B veri çekimi (Open-Meteo, yalnızca gerçek API verisi) ve Faktör C zaman/ölçek eşitleme.

Rastgele/sentetik veri üretilmez. Eksik adımlar yalnızca gerçek komşu değerlerden
PCHIP (rüzgâr/sıcaklık/basınç) ve berraklık indeksi (güneş) ile eşitlenir.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()


# --------------------------------------------------------------------------- HTTP
def om_get(url: str, params: dict, retries: int = 6) -> dict:
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 429:
                time.sleep(20 * (i + 1))
            elif r.status_code >= 500:
                time.sleep(3 * (i + 1))
            else:
                break
        except requests.RequestException as e:  # ağ hatası
            last = str(e)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"Open-Meteo isteği başarısız: {url} {last}")


def hourly_to_df(js: dict) -> pd.DataFrame:
    h = js["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    df.index = pd.to_datetime(h["time"], utc=True)
    df.index.name = "time"
    return df


def _fresh(path: Path, ttl_h: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_h * 3600


def cached(path: Path, ttl_h: float, fn):
    if _fresh(path, ttl_h):
        return pd.read_parquet(path)
    df = fn()
    df.to_parquet(path)
    return df


# --------------------------------------------------------------------------- değişkenler
def var_list(site: dict) -> list[str]:
    z = site.get("z_upper", 100)
    if site["type"] == "wind":
        return ["temperature_2m", "surface_pressure", "wind_speed_10m", f"wind_speed_{z}m",
                f"wind_direction_{z}m", "wind_gusts_10m", "cloud_cover"]
    return ["temperature_2m", "surface_pressure", "wind_speed_10m", "shortwave_radiation",
            "direct_normal_irradiance", "cloud_cover"]


def rename_std(df: pd.DataFrame, z_up: int) -> pd.DataFrame:
    m = {"temperature_2m": "temp", "surface_pressure": "pres", "wind_speed_10m": "v10",
         f"wind_speed_{z_up}m": "vup", f"wind_direction_{z_up}m": "dir", "wind_gusts_10m": "gust",
         "shortwave_radiation": "ghi", "direct_normal_irradiance": "dni", "cloud_cover": "cloud"}
    return df.rename(columns=m)


# --------------------------------------------------------------------------- çekim
def fetch_model(site: dict, model_names: list[str], cfg: dict, now: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """Tarihsel Tahmin API (eğitim penceresi) + canlı Forecast API (past_days=14, +16 gün)."""
    run = cfg["run"]
    vars_ = ",".join(var_list(site))
    z = site.get("z_upper", 100)
    hist_end = (now - pd.Timedelta(days=run["eval_days"] + 1)).date()
    hist_start = (now - pd.Timedelta(days=run["train_days"] + run["eval_days"])).date()
    last_err = None
    for name in model_names:
        try:
            live = hourly_to_df(om_get(cfg["api"]["forecast"], dict(
                latitude=site["lat"], longitude=site["lon"], hourly=vars_, models=name,
                past_days=run["eval_days"], forecast_days=run["forecast_days"],
                wind_speed_unit="ms", timezone="UTC")))
            live = rename_std(live, z)
            if live["v10"].notna().sum() < 24:
                raise RuntimeError("canlı veri boş")
            tag = f"{site['lat']}_{site['lon']}_{name}_{hist_start}_{hist_end}"
            hist = cached(CACHE / f"hist_{tag}.parquet", 1e9, lambda: rename_std(hourly_to_df(om_get(
                cfg["api"]["hist_forecast"], dict(
                    latitude=site["lat"], longitude=site["lon"], hourly=vars_, models=name,
                    start_date=str(hist_start), end_date=str(hist_end),
                    wind_speed_unit="ms", timezone="UTC"))), z))
            hist = hist[hist.index < live.index[0]]
            df = pd.concat([hist, live])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df, name
        except Exception as e:  # sıradaki yedek model adı
            last_err = e
    raise RuntimeError(f"Model alınamadı {model_names}: {last_err}")


def fetch_era5(site: dict, cfg: dict, now: pd.Timestamp) -> pd.DataFrame:
    """ERA5 (Archive API): yıllık parçalar; tamamlanmış yıllar kalıcı önbellekte."""
    vars_ = ",".join(var_list(site))
    # ERA5 her zaman 100 m rüzgâr verir (Hill of Towie için 80 m'ye alpha ile taşınır)
    if site["type"] == "wind":
        zu = site.get("z_upper", 100)
        vars_ = vars_.replace(f"wind_speed_{zu}m", "wind_speed_100m").replace(f"wind_direction_{zu}m", "wind_direction_100m")
    parts = []
    end_all = (now - pd.Timedelta(days=2)).date()
    for y in range(cfg["run"]["clim_start_year"], now.year + 1):
        s, e = pd.Timestamp(f"{y}-01-01").date(), min(pd.Timestamp(f"{y}-12-31").date(), end_all)
        if s > e:
            continue
        complete = y < now.year
        path = CACHE / f"era5_{site['lat']}_{site['lon']}_{y}.parquet"

        def get(s=s, e=e):
            return hourly_to_df(om_get(cfg["api"]["archive"], dict(
                latitude=site["lat"], longitude=site["lon"], hourly=vars_, models="era5",
                start_date=str(s), end_date=str(e), wind_speed_unit="ms", timezone="UTC")))
        parts.append(cached(path, 1e9 if complete else 6, get))
    df = pd.concat(parts)
    df = rename_std(df, 100)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_cams(site: dict, cfg: dict, now: pd.Timestamp) -> pd.DataFrame:
    """CAMS Air Quality: aerosol_optical_depth + dust (yüzey, µg/m³)."""
    run = cfg["run"]
    start = (now - pd.Timedelta(days=run["train_days"] + run["eval_days"] + 1)).date()
    end = (now + pd.Timedelta(days=4)).date()
    path = CACHE / f"cams_{site['lat']}_{site['lon']}.parquet"

    def get():
        parts = []
        s = pd.Timestamp(start)
        while s.date() <= end:  # 60 günlük parçalar
            e = min(s + pd.Timedelta(days=59), pd.Timestamp(end))
            parts.append(hourly_to_df(om_get(cfg["api"]["air_quality"], dict(
                latitude=site["lat"], longitude=site["lon"], hourly="aerosol_optical_depth,dust",
                start_date=str(s.date()), end_date=str(e.date()), timezone="UTC"))))
            s = e + pd.Timedelta(days=1)
        d = pd.concat(parts)
        return d[~d.index.duplicated(keep="last")].sort_index()
    df = cached(path, run["cache_ttl_hours"], get)
    return df.rename(columns={"aerosol_optical_depth": "aod"})


# --------------------------------------------------------------------------- eşitleme
def fill_pchip(s: pd.Series, max_gap: int = 6) -> pd.Series:
    """Geçerli noktalar arası ≤ max_gap saat olan boşlukları PCHIP ile doldurur (dışa uzatma yok)."""
    v = s.to_numpy(float)
    valid = ~np.isnan(v)
    if valid.sum() < 2:
        return s
    idx = np.arange(len(v))
    f = PchipInterpolator(idx[valid], v[valid], extrapolate=False)
    prev = pd.Series(np.where(valid, idx, np.nan)).ffill().to_numpy()
    nxt = pd.Series(np.where(valid, idx, np.nan)).bfill().to_numpy()
    ok = (~valid) & ((nxt - prev) <= max_gap)
    out = v.copy()
    out[ok] = f(idx[ok])
    return pd.Series(out, index=s.index)


def solar_geometry(index: pd.DatetimeIndex, site: dict) -> pd.DataFrame:
    import pvlib
    sp = pvlib.solarposition.get_solarposition(index, site["lat"], site["lon"])
    clr = pvlib.clearsky.haurwitz(sp["apparent_zenith"])["ghi"]
    return pd.DataFrame({"zenith": sp["apparent_zenith"], "azimuth": sp["azimuth"], "i_clr": clr}, index=index)


def harmonize(df: pd.DataFrame, site: dict, z_up: int | None = None, max_gap: int = 6) -> pd.DataFrame:
    """1 saatlik ortak ızgara; PCHIP + berraklık indeksi; dinamik alpha(t) ve hub rüzgârı."""
    idx = pd.date_range(df.index.min().floor("h"), df.index.max().ceil("h"), freq="1h", tz="UTC")
    d = df.reindex(idx).apply(pd.to_numeric, errors="coerce")
    for c in ("temp", "pres", "v10", "vup"):
        if c in d:
            d[c] = fill_pchip(d[c], max_gap)
    if "dir" in d:
        rad = np.deg2rad(d["dir"])
        sx, cy = fill_pchip(pd.Series(np.sin(rad), index=idx), max_gap), fill_pchip(pd.Series(np.cos(rad), index=idx), max_gap)
        d["dir"] = (np.rad2deg(np.arctan2(sx, cy)) + 360) % 360
    for c in ("gust", "cloud", "dni", "aod", "dust"):
        if c in d:
            d[c] = d[c].interpolate(limit=max_gap - 1, limit_area="inside")
    geo = solar_geometry(idx, site)
    d["zenith"], d["azimuth"], d["i_clr"] = geo["zenith"], geo["azimuth"], geo["i_clr"]
    if "ghi" in d:
        day = d["i_clr"] > 50
        kt = (d["ghi"] / d["i_clr"].where(day)).clip(0, 1.5)
        ktf = fill_pchip(kt, max_gap)
        ktf = ktf.ffill(limit=max_gap).bfill(limit=max_gap).where(day)
        covered = idx <= d["ghi"].last_valid_index() if d["ghi"].notna().any() else np.zeros(len(idx), bool)
        ghi = d["ghi"].copy()
        miss = ghi.isna() & day
        ghi[miss] = (ktf * d["i_clr"])[miss]
        ghi[ghi.isna() & ~day & covered] = 0.0
        d["ghi"] = ghi.clip(lower=0)
        d["kt"] = (d["ghi"] / d["i_clr"].where(day)).clip(0, 1.5)
        if "dni" in d:
            d["dni"] = d["dni"].clip(lower=0)
    if site["type"] == "wind":
        zu = z_up or site.get("z_upper", 100)
        alpha = np.log(d["vup"].clip(lower=0.05) / np.maximum(d["v10"], 0.5)) / np.log(zu / 10.0)
        d["alpha"] = alpha.clip(0.05, 0.45)
        d["vhub"] = d["vup"] * (site["hub_m"] / zu) ** d["alpha"]
    return d


# --------------------------------------------------------------------------- site yükleyici
def load_site(site_id: str, cfg: dict, now: pd.Timestamp, log=print) -> dict:
    site = cfg["sites"][site_id]
    out = {"models": {}, "model_names": {}}
    cams = fetch_cams(site, cfg, now)
    log(f"[{site_id}] CAMS: {cams['aod'].notna().sum()} saat AOD")
    aod_med = float(cams["aod"].dropna().median()) if cams["aod"].notna().any() else np.nan
    out["aod_median"] = aod_med
    out["dust_median"] = float(cams["dust"].dropna().median()) if cams["dust"].notna().any() else 0.0
    out["aod_p95"] = float(cams["aod"].dropna().quantile(0.95)) if cams["aod"].notna().any() else np.nan
    for key, names in cfg["models"].items():
        df, used = fetch_model(site, names, cfg, now)
        df = df.join(cams, how="left")
        out["models"][key] = harmonize(df, site)
        out["model_names"][key] = used
        log(f"[{site_id}] {key} ({used}): {len(df)} satır, v10 dolu {df['v10'].notna().sum()}")
    era = fetch_era5(site, cfg, now)
    out["era5_raw"] = era
    era_h = harmonize(era.join(cams, how="left"), site, z_up=100)
    out["era5"] = era_h
    log(f"[{site_id}] ERA5: {len(era)} saat, {era.index.min():%Y-%m-%d} - {era['v10'].last_valid_index():%Y-%m-%d}")
    return out
