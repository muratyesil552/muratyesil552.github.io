"""Faktör C: küçük ölçek fiziksel dönüşüm (rüzgâr ve güneş güç zinciri)."""
from __future__ import annotations

import numpy as np
import pandas as pd

R_D = 287.05
G = 9.80665


# --------------------------------------------------------------------------- yardımcılar
def fill_aero(df: pd.DataFrame, aod_med: float, dust_med: float) -> pd.DataFrame:
    """CAMS ufku dışındaki saatler için, gerçek CAMS kaydının medyanı kullanılır (işaretlenir)."""
    d = df.copy()
    d["aero_filled"] = d["aod"].isna() if "aod" in d else True
    for c, m in (("aod", aod_med), ("dust", dust_med)):
        d[c] = d[c].fillna(m) if c in d else m
    return d


def air_density(temp_c, pres_hpa, hub_m):
    """İdeal gaz yasası; yüzey basıncı hub yüksekliğine hipsometrik olarak taşınır."""
    T = temp_c + 273.15
    p_hub = pres_hpa * 100.0 * np.exp(-G * hub_m / (R_D * T))
    return p_hub / (R_D * T)


def power_curve(v, cap, vin, vr, vout, cutout=True):
    v = np.asarray(v, float)
    x = np.clip((v ** 3 - vin ** 3) / (vr ** 3 - vin ** 3), 0.0, 1.0)
    p = cap * x
    p = np.where(v < vin, 0.0, p)
    if cutout:
        p = np.where(v >= vout, 0.0, p)
    return np.where(np.isnan(v), np.nan, p)


def wake_loss(direction, v, site, wcfg):
    """Jensen/Gauss tipi analitik iz kaybı: ana dizilim ekseninde %max, çapraz yönde %min;
    itki katsayısı rüzgâr nominal hızı aşınca azalır (Ct ~ (vr/v)^3)."""
    c = np.cos(np.deg2rad(np.asarray(direction, float) - site["main_axis_deg"])) ** 2
    base = wcfg["wake_min"] + (wcfg["wake_max"] - wcfg["wake_min"]) * c
    vr = site["rated_speed"]
    ct = np.where(v <= vr, 1.0, (vr / np.maximum(v, 1e-6)) ** 3)
    return np.where(np.isnan(base), wcfg["wake_min"], base) * ct


def dust_soiling(dust, scfg):
    return np.minimum(scfg["dust_soiling_max"], scfg["dust_soiling_per_ugm3"] * np.maximum(dust, 0.0))


def storm_mask(v, gust, tcfg):
    m = np.asarray(v, float) >= tcfg["cutout_speed"]
    if gust is not None:
        g = np.asarray(gust, float)
        m = m | (np.nan_to_num(g, nan=0.0) >= tcfg["cutout_gust"])
    return m


# --------------------------------------------------------------------------- rüzgâr
def wind_power(df: pd.DataFrame, site: dict, cfg: dict, mode: str = "full") -> pd.Series:
    """mode='raw': yalnız standart güç eğrisi (yoğunluk/iz/toz/hamle yok).
       mode='full': yoğunluk + iz kaybı + toz + (hub>=25 m/s veya hamle>=28 m/s) fırtına kesmesi."""
    cap, vin, vr, vout = site["capacity_mw"], site["cut_in"], site["rated_speed"], site["cut_out"]
    v = df["vhub"].to_numpy(float)
    if mode == "raw":
        p = power_curve(v, cap, vin, vr, vout)
        return pd.Series(p, index=df.index)
    rho = air_density(df["temp"].to_numpy(float), df["pres"].to_numpy(float), site["hub_m"])
    veq = v * (rho / cfg["wind_physics"]["rho_ref"]) ** (1 / 3)
    p = power_curve(veq, cap, vin, vr, vout, cutout=False)
    p = p * (1 - wake_loss(df["dir"].to_numpy(float), v, site, cfg["wind_physics"]))
    p = p * (1 - dust_soiling(df["dust"].to_numpy(float), cfg["solar_physics"]))
    gust = df["gust"].to_numpy(float) if "gust" in df else None
    storm = storm_mask(v, gust, cfg["thresholds"]) | (v >= vout)
    p = np.where(storm, 0.0, p)
    p = np.where(np.isnan(v) | np.isnan(rho), np.nan, p)
    return pd.Series(np.clip(p, 0, cap), index=df.index)


# --------------------------------------------------------------------------- güneş
def poa_irradiance(df: pd.DataFrame, site: dict, cfg: dict, ghi: pd.Series | None = None) -> pd.Series:
    import pvlib
    g = df["ghi"] if ghi is None else ghi
    doy = df.index.dayofyear
    extra = pvlib.irradiance.get_extra_radiation(df.index)
    er = pvlib.irradiance.erbs(g.clip(lower=0), df["zenith"], doy)
    tot = pvlib.irradiance.get_total_irradiance(
        site["tilt"], site["azimuth"], df["zenith"], df["azimuth"],
        dni=pd.Series(np.asarray(er["dni"], float), index=df.index).fillna(0), ghi=g,
        dhi=pd.Series(np.asarray(er["dhi"], float), index=df.index).fillna(0), dni_extra=extra,
        albedo=cfg["solar_physics"]["albedo"], model="isotropic")
    return tot["poa_global"].where(g.notna())


def solar_power(df: pd.DataFrame, site: dict, cfg: dict, mode: str = "full", aod_med: float = 0.1,
                parts: bool = False):
    sp = cfg["solar_physics"]
    cap = site["capacity_mw"]
    poa0 = poa_irradiance(df, site, cfg)
    raw = cap * sp["system_efficiency"] * poa0 / 1000.0
    if mode == "raw":
        return raw.clip(0, cap)
    atten = np.exp(-sp["aod_atten_k"] * np.maximum(df["aod"] - aod_med, 0.0))
    poa = poa0 * atten
    tcell = df["temp"] + poa / (sp["faiman_u0"] + sp["faiman_u1"] * df["v10"].clip(lower=0))
    eff_t = 1 + sp["gamma_pmp"] * (tcell - 25.0)
    soil = dust_soiling(df["dust"], sp)
    full = cap * sp["system_efficiency"] * poa / 1000.0 * eff_t * (1 - soil)
    full = full.clip(0, cap)
    if parts:
        return {"raw": raw, "poa0": poa0, "atten": atten, "tcell": tcell, "eff_t": eff_t, "soil": soil, "full": full}
    return full


def site_power(df, site, cfg, mode="full", aod_med=0.1):
    if site["type"] == "wind":
        return wind_power(df, site, cfg, mode)
    return solar_power(df, site, cfg, mode, aod_med)


# --------------------------------------------------------------------------- orta ölçek birleşimi
def blend_meso(ifs: pd.DataFrame, icon: pd.DataFrame, site: dict) -> pd.DataFrame:
    """M3/M4 girdisi: IFS ve ICON girdilerinin ortalaması; ICON ufku dışında yalnız IFS."""
    cols = [c for c in ("temp", "pres", "v10", "vup", "gust", "ghi", "dni", "cloud", "kt", "alpha", "vhub") if c in ifs]
    idx = ifs.index.union(icon.index)
    a, b = ifs.reindex(idx), icon.reindex(idx)
    out = a.copy()
    for c in cols:
        out[c] = pd.concat([a[c], b[c]], axis=1).mean(axis=1, skipna=True)
    if "dir" in a:
        rad_a, rad_b = np.deg2rad(a["dir"]), np.deg2rad(b["dir"])
        sx = pd.concat([np.sin(rad_a), np.sin(rad_b)], axis=1).mean(axis=1, skipna=True)
        cy = pd.concat([np.cos(rad_a), np.cos(rad_b)], axis=1).mean(axis=1, skipna=True)
        out["dir"] = (np.rad2deg(np.arctan2(sx, cy)) + 360) % 360
    out["icon_used"] = b["v10"].notna() & a["v10"].notna()
    if site["type"] == "wind":
        # blend sonrası hub rüzgârı: bileşenlerin hub rüzgârlarının ortalaması (kesme üssü ayrıca ortalanır)
        out["vhub"] = pd.concat([a["vhub"], b["vhub"]], axis=1).mean(axis=1, skipna=True)
        out["alpha"] = pd.concat([a["alpha"], b["alpha"]], axis=1).mean(axis=1, skipna=True)
    return out


# --------------------------------------------------------------------------- kayıp ağacı (iklimsel)
def wind_loss_tree(df: pd.DataFrame, site: dict, cfg: dict) -> dict:
    """Saatlik ERA5 girdisinden ardışık kayıp bileşenleri (MWh toplamları)."""
    cap, vin, vr, vout = site["capacity_mw"], site["cut_in"], site["rated_speed"], site["cut_out"]
    v = df["vhub"].to_numpy(float)
    gross = power_curve(v, cap, vin, vr, vout, cutout=False)
    storm = storm_mask(v, df["gust"].to_numpy(float), cfg["thresholds"]) | (v >= vout)
    p1 = np.where(storm, 0.0, gross)
    rho = air_density(df["temp"].to_numpy(float), df["pres"].to_numpy(float), site["hub_m"])
    veq = v * (rho / cfg["wind_physics"]["rho_ref"]) ** (1 / 3)
    p2 = np.where(storm, 0.0, power_curve(veq, cap, vin, vr, vout, cutout=False))
    p3 = p2 * (1 - wake_loss(df["dir"].to_numpy(float), v, site, cfg["wind_physics"]))
    p4 = p3 * (1 - dust_soiling(df["dust"].to_numpy(float), cfg["solar_physics"]))
    s = lambda x: float(np.nansum(x))
    return {"gross": s(gross), "cutout": s(gross) - s(p1), "density": s(p1) - s(p2),
            "wake": s(p2) - s(p3), "dust": s(p3) - s(p4), "net": s(p4)}


def solar_loss_tree(df: pd.DataFrame, site: dict, cfg: dict, aod_med: float) -> dict:
    parts = solar_power(df, site, cfg, "full", aod_med, parts=True)
    cap, sp = site["capacity_mw"], cfg["solar_physics"]
    ideal = cap * sp["system_efficiency"] * parts["poa0"] / 1000.0
    after_atten = cap * sp["system_efficiency"] * (parts["poa0"] * parts["atten"]) / 1000.0
    after_temp = after_atten * parts["eff_t"]
    net = after_temp * (1 - parts["soil"])
    s = lambda x: float(np.nansum(np.clip(x, 0, cap)))
    return {"gross": s(ideal), "aod": s(ideal) - s(after_atten), "temperature": s(after_atten) - s(after_temp),
            "dust": s(after_temp) - s(net), "net": s(net)}
