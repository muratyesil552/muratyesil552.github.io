"""Open-Meteo canlı yetenek yoklaması: hangi model hangi değişkeni gerçekten veriyor?"""
import requests, json, sys

URL = "https://api.open-meteo.com/v1/forecast"
LAT, LON = 40.15, 26.40
MODELS = ["gfs_seamless", "ecmwf_aifs025_single", "ecmwf_ifs025", "ecmwf_ifs", "icon_seamless", "icon_eu"]
VARS = ["temperature_2m", "surface_pressure", "wind_speed_10m", "wind_speed_80m", "wind_speed_100m",
        "wind_speed_120m", "wind_direction_10m", "wind_direction_80m", "wind_direction_100m",
        "wind_gusts_10m", "shortwave_radiation", "direct_normal_irradiance", "cloud_cover"]

for m in MODELS:
    r = requests.get(URL, params={"latitude": LAT, "longitude": LON, "hourly": ",".join(VARS),
                                  "models": m, "forecast_days": 16, "wind_speed_unit": "ms",
                                  "timezone": "UTC"}, timeout=60)
    if r.status_code != 200:
        print(m, "HTTP", r.status_code, r.text[:200]); continue
    h = r.json()["hourly"]
    n = len(h["time"])
    out = {}
    for v in VARS:
        vals = h.get(v)
        if vals is None:
            out[v] = "yok"; continue
        ok = [x for x in vals if x is not None]
        out[v] = f"{len(ok)}/{n}"
    print(m, n, json.dumps(out, ensure_ascii=False))
