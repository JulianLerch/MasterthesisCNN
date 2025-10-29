# ⚡ SCHNELLSTART-ANLEITUNG

**TIFF Simulator V3.0 - In 3 Minuten starten!**

---

## 🚀 METHODE 1: GUI (Einfach)

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. GUI starten
python START.py

# oder direkt:
python tiff_simulator_gui.py
```

**✅ Das war's!** Die GUI führt dich durch alles.

---

## 🐍 METHODE 2: Python Code

```python
from tiff_simulator_v3 import TDI_PRESET, TIFFSimulator, save_tiff

# Simulator erstellen
sim = TIFFSimulator(detector=TDI_PRESET, mode="polyzeit", t_poly_min=60.0)

# TIFF generieren
tiff = sim.generate_tiff(image_size=(128, 128), num_spots=10, num_frames=100)

# Speichern
save_tiff("simulation.tif", tiff)
```

---

## 📓 METHODE 3: Jupyter Notebook (für Anfänger!)

```bash
# 1. Jupyter starten
jupyter notebook

# 2. Öffne: JUPYTER_TUTORIAL.ipynb

# 3. Führe Zellen aus (Shift+Enter)
```

**✅ Vollständiges Tutorial** mit Beispielen und Visualisierungen!

---

## 🔄 METHODE 4: Batch-Modus

```bash
# Quick Test (3 TIFFs, ~2 Min)
python batch_simulator.py --preset quick --output ./test

# Masterthesis (60+ TIFFs, ~45 Min)
python batch_simulator.py --preset thesis --output ./thesis_data

# Publication Quality (30 TIFFs, ~2 Std)
python batch_simulator.py --preset publication --output ./publication
```

---

## 📋 DATEI-ÜBERSICHT

```
tiff_simulator_complete/
│
├── 📄 START.py                    ← Starte HIER! (GUI mit Checks)
├── 🎮 tiff_simulator_gui.py       ← GUI direkt
├── 🔬 tiff_simulator_v3.py        ← Core Backend
├── 📋 metadata_exporter.py        ← Metadata-System
├── 🔄 batch_simulator.py          ← Batch-Modus
│
├── 📖 README.md                   ← Vollständige Doku
├── ⚡ QUICKSTART.md               ← Diese Datei
├── 📓 JUPYTER_TUTORIAL.ipynb      ← Jupyter Tutorial
│
└── 📦 requirements.txt            ← Dependencies
```

---

## ❓ PROBLEME?

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "GUI startet nicht"
- **Windows/macOS**: tkinter sollte built-in sein
- **Linux**: `sudo apt-get install python3-tk`

### "Simulation zu langsam"
Nutze kleinere Parameter:
- `image_size=(64, 64)` statt (256, 256)
- `num_spots=5` statt 50
- `num_frames=50` statt 500

---

## 💡 ERSTE SCHRITTE

1. **Test-Simulation** (GUI):
   - Starte `START.py`
   - Wähle "Quick Test" Preset
   - Klick "🚀 SIMULATION STARTEN"
   - ✅ Fertig in ~1 Minute!

2. **Jupyter Tutorial** (für Code-Anfänger):
   - Öffne `JUPYTER_TUTORIAL.ipynb`
   - Befolge Schritt-für-Schritt Anleitung
   - Visualisiere Ergebnisse direkt!

3. **Lies README.md** für:
   - Wissenschaftliche Details
   - Physikalisches Modell
   - Parameter-Referenz
   - Datenanalyse-Tipps

---

## 🎯 EMPFOHLENE WORKFLOWS

### Für schnelle Tests:
```python
image_size = (64, 64)
num_spots = 5
num_frames = 30
→ ~10 Sekunden
```

### Für realistische Daten:
```python
image_size = (128, 128)
num_spots = 15
num_frames = 200
→ ~1 Minute
```

### Für Publication-Quality:
```python
image_size = (256, 256)
num_spots = 50
num_frames = 500
→ ~5-10 Minuten
```

---

## 📚 WEITERE HILFE

- **README.md**: Vollständige wissenschaftliche Dokumentation
- **JUPYTER_TUTORIAL.ipynb**: Schritt-für-Schritt Code-Beispiele
- **Code-Kommentare**: Alle Funktionen sind dokumentiert

---

**Los geht's! 🚀**

Starte mit `python START.py` und erstelle deine erste Simulation! 🔬✨
