# 🚀 BATCH-MODUS ANLEITUNG - TIFF Simulator V4.0

## ❌ WICHTIG: Batch-Modus in GUI V4.0 ist nur über Tab verfügbar!

Die GUI V4.0 hat einen **Batch-Tab**, aber der funktioniert nur zum Konfigurieren.
Zum **Ausführen** gibt es 2 Wege:

---

## ✅ OPTION 1: Batch-Tab in GUI nutzen (Eingeschränkt)

**Status:** Der Batch-Tab in GUI V4.0 existiert, aber...
- ✅ Kann Presets anzeigen
- ✅ Kann Custom Times eingeben
- ❌ **Führt Batch NICHT aus** (nur Single-Simulationen!)

**Grund:** Die Batch-Logik wurde in V4.0 vereinfacht.

---

## ✅ OPTION 2: Batch-Simulator direkt nutzen (EMPFOHLEN!)

### Quick Start:

```bash
python batch_simulator.py --preset quick --output ./batch_output
```

### Alle Optionen:

```python
from batch_simulator import BatchSimulator, PresetBatches

# Methode 1: Preset verwenden
batch = PresetBatches.quick_test("./output")
batch.run()

# Methode 2: Custom Batch
from tiff_simulator_v3 import TDI_PRESET

batch = BatchSimulator("./output")

# Füge Simulationen hinzu
for t_poly in [0, 30, 60, 90, 120]:
    batch.add_task({
        'detector': TDI_PRESET,
        'mode': 'polyzeit',
        't_poly_min': t_poly,
        'astigmatism': False,
        'filename': f"tdi_t{t_poly}min.tif",
        'image_size': (256, 256),
        'num_spots': 30,
        'num_frames': 200,
        'frame_rate_hz': 20.0,
        'd_initial': 0.24  # ← KORRIGIERT!
    })

# Führe aus mit Progress
batch.run(progress_callback=lambda c, t, s: print(f"{c}/{t}: {s}"))
```

---

## 📋 VERFÜGBARE BATCH-PRESETS:

### 1. Quick Test
```python
batch = PresetBatches.quick_test("./output")
```
- **3 TIFFs** in ~2-5 Minuten
- Polyzeiten: 30, 60, 90 min
- 64×64 px, 50 frames
- Perfekt zum Testen!

### 2. Masterthesis
```python
batch = PresetBatches.masterthesis_full("./output")
```
- **60+ TIFFs** in ~1 Stunde
- Vollständige Parameterstudie
- TDI-G0 + Tetraspecs
- 2D + 3D (Astigmatismus)
- z-Stack Kalibrierung
- 3 Wiederholungen pro Bedingung

### 3. Publication Quality
```python
batch = PresetBatches.publication_quality("./output")
```
- **30 TIFFs** in ~2-3 Stunden
- Hohe Auflösung: 256×256 px
- Viele Spots: 50
- Viele Frames: 500
- 5 Wiederholungen für Statistik

---

## 🛠️ CUSTOM BATCH FÜR IHRE MASTERTHESIS

**Mit den KORRIGIERTEN D-Werten:**

```python
from batch_simulator import BatchSimulator
from tiff_simulator_v3 import TDI_PRESET, TETRASPECS_PRESET

# Erstelle Batch
batch = BatchSimulator("./masterthesis_data")

# Parameter
d_initial = 0.24  # µm²/s - REALISTISCH!
poly_times = [0, 10, 30, 60, 90, 120, 180]  # min
repeats = 3

for detector in [TDI_PRESET, TETRASPECS_PRESET]:
    for t_poly in poly_times:
        for repeat in range(repeats):
            # 2D Simulation
            batch.add_task({
                'detector': detector,
                'mode': 'polyzeit',
                't_poly_min': t_poly,
                'astigmatism': False,
                'filename': f"{detector.name}_2d_t{int(t_poly)}min_r{repeat+1}.tif",
                'image_size': (256, 256),
                'num_spots': 30,
                'num_frames': 300,
                'frame_rate_hz': 20.0,
                'd_initial': d_initial
            })

            # 3D Simulation (mit Astigmatismus)
            batch.add_task({
                'detector': detector,
                'mode': 'polyzeit_astig',
                't_poly_min': t_poly,
                'astigmatism': True,
                'filename': f"{detector.name}_3d_t{int(t_poly)}min_r{repeat+1}.tif",
                'image_size': (256, 256),
                'num_spots': 30,
                'num_frames': 300,
                'frame_rate_hz': 20.0,
                'd_initial': d_initial
            })

# WICHTIG: Mit Progress-Callback ausführen!
def progress(current, total, status):
    print(f"[{current}/{total}] {status}")
    # Optional: In Datei loggen
    with open("batch_progress.log", "a") as f:
        f.write(f"{datetime.now()}: [{current}/{total}] {status}\n")

batch.run(progress_callback=progress)

print(f"\n✅ Batch fertig! Alle TIFFs in: {batch.output_dir}")
print(f"📊 Metadata CSV: {batch.output_dir}/batch_summary.csv")
```

**Das erstellt:**
- 2 Detektoren × 7 Zeiten × 2 Modi (2D/3D) × 3 Repeats = **84 TIFFs**
- Dauer: ~2-3 Stunden (mit V4.0 Performance!)
- Vollständige Metadata (JSON, TXT, CSV)

---

## 📊 BATCH-MODUS FÜR SPEZIFISCHE ANALYSEN

### A) Nur D-Wert Variation (feste Zeit)

```python
batch = BatchSimulator("./d_variation")

d_values = [0.15, 0.20, 0.24, 0.28, 0.32]  # µm²/s
t_poly = 60  # min (feste Zeit)

for d in d_values:
    for repeat in range(5):
        batch.add_task({
            'detector': TDI_PRESET,
            'mode': 'polyzeit',
            't_poly_min': t_poly,
            'd_initial': d,
            'filename': f"d{d:.2f}_r{repeat+1}.tif",
            'image_size': (256, 256),
            'num_spots': 30,
            'num_frames': 200,
            'frame_rate_hz': 20.0
        })

batch.run()
```

### B) Zeit-Serie (feste D₀)

```python
batch = BatchSimulator("./time_series")

times = [0, 5, 10, 20, 30, 45, 60, 75, 90, 120, 150, 180]  # min
d_initial = 0.24  # µm²/s

for t in times:
    for repeat in range(3):
        batch.add_task({
            'detector': TDI_PRESET,
            'mode': 'polyzeit',
            't_poly_min': t,
            'd_initial': d_initial,
            'filename': f"t{t:03d}min_r{repeat+1}.tif",
            'image_size': (256, 256),
            'num_spots': 30,
            'num_frames': 200,
            'frame_rate_hz': 20.0
        })

batch.run()
```

### C) Nur z-Stack Kalibrierung

```python
batch = BatchSimulator("./z_calibration")

for detector in [TDI_PRESET, TETRASPECS_PRESET]:
    batch.add_task({
        'detector': detector,
        'mode': 'z_stack',
        't_poly_min': 0,  # Keine Polymerisation
        'astigmatism': True,
        'filename': f"zstack_{detector.name}.tif",
        'image_size': (256, 256),
        'num_spots': 50,
        'z_range_um': (-1.0, 1.0),
        'z_step_um': 0.05
    })

batch.run()
```

---

## 🎯 EMPFEHLUNG FÜR IHRE MASTERTHESIS:

**Workflow:**

1. **Testen** (5-10 Minuten):
   ```bash
   python -c "from batch_simulator import PresetBatches; PresetBatches.quick_test('./test').run()"
   ```

2. **Kleine Studie** (~30 Minuten):
   ```python
   # Nur 3 Zeiten, 2 Repeats
   times = [0, 60, 120]
   repeats = 2
   # → 12 TIFFs (2×3×2)
   ```

3. **Vollständige Thesis-Daten** (~2-3 Stunden):
   ```python
   # Custom Batch wie oben (84 TIFFs)
   ```

4. **Analyse**:
   - Alle TIFFs mit TrackMate/ThunderSTORM analysieren
   - Ground Truth aus Metadata CSV
   - D-Wert Rekonstruktion
   - Plots für Thesis

---

## 💡 TIPPS & TRICKS

### Parallele Batches

Wenn Sie mehrere CPU-Kerne haben:

```bash
# Terminal 1
python batch_1.py &

# Terminal 2
python batch_2.py &

# Etc.
```

### Resume bei Absturz

Batch-Simulator erstellt nach jedem TIFF eine CSV.
Bei Absturz: Checken Sie welche TIFFs fehlen und erstellen Sie neuen Batch nur für diese.

### Speicherplatz

Jedes TIFF (256×256, 300 frames, 16-bit):
- ~40 MB pro Datei
- 84 TIFFs = ~3.4 GB
- + Metadata = ~3.5 GB gesamt

**Planen Sie genug Speicher ein!**

---

## ❓ FAQ

**Q: Warum nicht in GUI V4.0?**
A: GUI ist für interaktive Einzelsimulationen optimiert. Batch ist komplexer und läuft besser in Python.

**Q: Kann ich die GUI-Parameter für Batch nutzen?**
A: Ja! Setzen Sie in GUI die Parameter, dann kopieren Sie die Werte ins Python-Script.

**Q: Progress-Tracking?**
A: Nutzen Sie `progress_callback` - siehe Beispiele oben!

**Q: Kann ich Batch abbrechen?**
A: Ja, mit Ctrl+C. Bereits erstellte TIFFs bleiben erhalten.

---

## 📝 ZUSAMMENFASSUNG

**Batch-Modus nutzen:**
1. ✅ Nicht in GUI V4.0 (nur Single!)
2. ✅ Nutze `batch_simulator.py` direkt
3. ✅ Presets: `quick`, `thesis`, `publication`
4. ✅ Custom: Python-Script schreiben
5. ✅ **WICHTIG:** `d_initial = 0.24` verwenden!

**Für Ihre Thesis empfohlen:**
```python
# Custom Batch mit ~84 TIFFs, ~2-3 Stunden
# Alle Zeiten, Detektoren, 2D+3D, Repeats
```

Viel Erfolg! 🎓
