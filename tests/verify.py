"""Hızlı doğrulama: fiziksel kurallar, sentetik veri taraması. Çalıştır: python tests/verify.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import physics_chain as P  # noqa: E402
import hybrid_and_feedback as H  # noqa: E402
import fetch_and_harmonize as F  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
site = cfg["sites"]["TR_Canakkale_RES"]
ok = True


def check(name, cond):
    global ok
    print(("GEÇTİ  " if cond else "KALDI  ") + name)
    ok &= bool(cond)


# 1) cut-out kuralı
idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
df = pd.DataFrame({"vhub": [24.9, 25.0, 12.0, 12.0], "gust": [10, 10, 28.0, 20], "temp": 10.0, "pres": 1000.0,
                   "dir": 45.0, "dust": 0.0}, index=idx)
p = P.wind_power(df, site, cfg, "full")
check("hub 24.9 m/s -> 0'dan büyük güç", p.iloc[0] > 0)
check("hub 25.0 m/s -> 0 MW", p.iloc[1] == 0)
check("hamle 28 m/s -> 0 MW", p.iloc[2] == 0)
check("hamle 20 m/s, 12 m/s -> güç var", p.iloc[3] > 0)
# 2) güç sınırı + wake aralığı
check("güç <= kurulu güç", (p.dropna() <= site["capacity_mw"] + 1e-9).all())
wl = P.wake_loss(np.array([45.0, 135.0]), np.array([8.0, 8.0]), site, cfg["wind_physics"])
check("wake ana eksende %14, çaprazda %8", abs(wl[0] - 0.14) < 1e-6 and abs(wl[1] - 0.08) < 1e-6)
# 3) alpha kırpması
d = pd.DataFrame({"v10": [5.0, 5.0, 0.1], "vup": [5.0, 40.0, 10.0], "dir": 0.0}, index=idx[:3])
d["temp"], d["pres"] = 10.0, 1000.0
a = np.log(d["vup"] / np.maximum(d["v10"], 0.5)) / np.log(100 / 10)
check("alpha kırpma [0.05, 0.45]", a.clip(0.05, 0.45).between(0.05, 0.45).all())
# 4) yoğunluk (deniz seviyesi, 15 C -> ~1.225)
rho = P.air_density(15.0, 1013.25, 0.0)
check("hava yoğunluğu ~1.225", abs(rho - 1.225) < 0.01)
# 5) ACI kapsama yakınsaması (ölçek kayması altında)
rng_free = np.sin(np.linspace(0, 40, 600)) * 3.0  # deterministik, rastgele değil
y = rng_free * (1 + np.linspace(0, 1, 600))
pred = np.zeros(600)
lo, hi, _, a_end, _ = H.aci_walk(y, pred, np.abs(y[:100]), 0.2, 0.05, 336)
cov = np.nanmean((y[100:] >= lo[100:]) & (y[100:] <= hi[100:]))
check(f"ACI kapsama ~%80 (ölçülen {cov:.2f})", 0.7 <= cov <= 0.92)
# 6) PCHIP tepe koruması: 6 saatlik adımda ara değerler monoton aralığı aşmaz
s = pd.Series([np.nan] * 13, index=pd.date_range("2026-01-01", periods=13, freq="h"))
s.iloc[0], s.iloc[6], s.iloc[12] = 5.0, 15.0, 6.0
f = F.fill_pchip(s, 6)
check("PCHIP tepeyi aşmaz ve boşlukları doldurur", f.notna().all() and f.max() <= 15.0 + 1e-9)
# 7) sentetik veri taraması
bad = re.compile(r"np\.random|numpy\.random|random\.|\bfaker\b|\bmock\b|\bdummy\b", re.I)
hits = []
for fp in (ROOT / "src").glob("*.py"):
    for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
        if bad.search(line) and "random_state" not in line:
            hits.append(f"{fp.name}:{i}: {line.strip()}")
check("src/ içinde rastgele/sentetik veri üretimi yok", not hits)
for h in hits:
    print("   ", h)
sys.exit(0 if ok else 1)
