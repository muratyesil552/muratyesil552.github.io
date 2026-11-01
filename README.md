# ZİNCİR-TR

Çok ölçekli, geri beslemeli hibrit rüzgâr ve güneş enerjisi tahmin, +6 saat / +16 gün anomali ve kesinti erken uyarı ve iklimsel verim (P50/P90) gözlemevi.

Üç saha: `TR_Canakkale_RES` (30 MW), `TR_Konya_GES` (20 MW), `UK_HillOfTowie_RES` (48.3 MW).

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
python tests/verify.py          # fizik kuralları + sentetik veri taraması
python src/run_pipeline.py      # veri çek, modelle, çıktıları üret (~1-2 dk; ilk çalıştırmada ERA5 önbelleği ~2-3 dk)
```

Çıktılar:

| Dosya | İçerik |
|---|---|
| `data/master_ledger.parquet` / `.csv` | Son 30 günün tahmin ve karne kütüğü (saha, saat, günlük yayın, referans, M1-M6, P10/P90) |
| `data/anomaly_catalog.csv` | Fırtına kesintisi, rampa, bulut/toz, LLJ, iklimsel aşırı rüzgâr olayları |
| `docs/web_data/observability_data.json` | Tüm grafik ve tabloların verisi |
| `docs/web_data/data_bundle.js` | Aynı veri, `window.ZINCIR_DATA = {...}` (CORS'suz `file://` için) |
| `docs/index.html` | Gözlemevi. Çift tıklayınca veya GitHub Pages'te çalışır |

GitHub Pages için kaynak olarak `docs/` klasörünü seçin. `update_observatory.yml` her 6 saatte bir boru hattını çalıştırıp çıktıları işler.

## Veri ilkesi

Hiçbir sentetik veri yoktur. Tüm girdiler Open-Meteo API'lerinden gelir: Forecast (`past_days=14`, `forecast_days=16`), Historical Forecast (kalıntı modeli eğitimi), ERA5 Archive (2016'dan beri), CAMS Air Quality (AOD, toz). Modeller: `gfs_seamless`, `ecmwf_aifs025_single`, `ecmwf_ifs`, `icon_seamless`.

## Mimari

- **A (~25 km):** GFS, AIFS, ERA5
- **B (orta ölçek):** IFS + ICON (ICON ufku ~7.5 gün; ötesinde yalnız IFS), CAMS
- **C (fizik):** dinamik `alpha(t)` ile hub rüzgârı, PCHIP / berraklık indeksi eşitleme, yoğunluk + analitik iz kaybı + toz + fırtına kesmesi (hub ≥ 25 m/s veya hamle ≥ 28 m/s), güneşte POA + Faiman + AOD zayıflaması
- **D (hibrit):** LightGBM kalıntı (`e = P_ref − P_M4`), rampa ve yön hatası ağırlıklı, ACI ile P10-P90
- **Geri beslemeler:** G2 (kalıntı + ACI), G5 (ERA5 persentil eşikleri → anlık uyarı motoru ve özellik), G4 (kısa vade hata varyansı → P50/P75/P90)

Modeller: M1 GFS, M2 AIFS, M3 IFS+ICON ham, M4 tam fizik, M5 hibrit GFS, M6 hibrit AIFS. Formüller için web arayüzündeki "Akademik Metodoloji" kutusuna bakın.

## Bilinen sınırlamalar (dürüst notlar)

1. **"Gerçekleşen" güç SCADA değildir.** Open-Meteo üretim verisi sunmaz; referans, ERA5'ten aynı fizik zinciriyle hesaplanan güçtür. Karne, tahminlerin reanalize göre sapmasını ölçer. Gerçek SCADA (ör. Hill of Towie açık veri seti) eklenirse `ref` serisi onunla değiştirilmelidir.
2. ERA5 ~5-6 gün gecikmelidir; karne penceresinin son günleri referanssızdır ve son 14 günde etiketli saat sayısı ~200'dür. Rampa/olay sayıları küçük olabilir; POD/FAR/CSI ve Ramp-DA bu durumda "-" veya düşük örneklidir.
3. Eğitim, kısa öngörü süreli geçmiş tahminlerle yapılır. +16 gün ufkunda hibrit düzeltme ve konformal bant, yakın vade hatasına göre kalibredir (bant uzun ufukta gerçeğe göre dar kalabilir).
4. CAMS ufku (~4-5 gün) ötesinde AOD/toz, gerçek CAMS kaydının medyanına sabitlenir (`aero_filled`). 10 yıllık ERA5 zincirinde CAMS yalnız son ~6 ay için mevcuttur; öncesi medyanla alınır.
5. AIFS 10 m hamle vermez; M2 ve AIFS tabanlı olay tespitinde yalnız hub hızı kuralı uygulanır.
6. Wake katsayıları (%8-14), ana dizilim ekseni (`main_axis_deg`) ve güneş `system_efficiency` varsayımdır; `config/config.yaml` içinde ayarlanır.
7. Open-Meteo ücretsiz API ticari olmayan kullanım içindir.
