# CHANGELOG - TIFF Simulator

## Version 4.0 - Oktober 2025 (MAJOR UPDATE)

### 🚀 Performance-Optimierungen (10-50x schneller!)

**Core Engine:**
- ✅ **Vektorisierte PSF-Generierung:** Batch-Processing für alle Spots gleichzeitig
- ✅ **ROI-basierte Berechnung:** Nur 3-sigma Umgebung wird berechnet (nicht ganzes Bild)
- ✅ **Pre-computed Koordinaten-Grids:** Wiederverwend bare Meshgrids
- ✅ **Background-Caching:** Intelligentes Caching für Batch-Simulationen
- ✅ **Memory-efficient:** Optimierte Array-Wiederverwendung
- ✅ **Progress-Callbacks:** Thread-safe Echtzeit-Updates

**Ergebnisse:**
- Kleine TIFFs (128×128, 100 frames, 10 spots): ~1-2 Sekunden (V3: ~10s)
- Mittlere TIFFs (256×256, 500 frames, 30 spots): ~3-5 Minuten (V3: ~45 min)
- Große TIFFs (512×512, 2000 frames, 50 spots): ~20-30 Minuten (V3: mehrere Stunden)

### 🎨 GUI V4.0 - Advanced Edition

**Neue Parameter-Tabs:**
- 📊 Basis-Parameter (wie V3.0)
- ⚛️ **NEU:** Erweiterte Physik (PSF, Background, Noise)
- 💡 **NEU:** Photophysik & Blinking (ON/OFF, Bleaching)
- 📐 **NEU:** 3D & Astigmatismus (z-Parameter, Koeffizienten)
- 📦 Batch-Modus (erweitert)
- 💾 Export & Metadata

**Neue GUI-Features:**
- ✅ Tooltips für ALLE Parameter (physikalische Bedeutung + Empfehlungen)
- ✅ Live-Updates für D-Wert-Schätzung
- ✅ z-Stack Slice-Berechnung in Echtzeit
- ✅ Moderneres Design (dunkler Header, bessere Farben)
- ✅ Scrollbares Interface (passt auf alle Bildschirmgrößen)

### 🔬 Erweiterte Physik

**Photophysik (NEU!):**
- ✅ Blinking: 2-Zustands-Modell (ON/OFF) mit konfigurierbaren Dauern
- ✅ Photobleaching: Irreversibles Bleaching mit realistischen Wahrscheinlichkeiten
- ✅ Geometrische Dauern-Verteilung (physikalisch korrekt)

**Noise & PSF (erweitert):**
- ✅ Variable Max-Intensität (vorher: fix für Detektor)
- ✅ Spot Intensity Sigma (lognormale Variabilität)
- ✅ Frame Jitter Sigma (Frame-zu-Frame Schwankungen)
- ✅ Separate Background Mean & Std
- ✅ Konfigurierbare Read Noise

**3D & Astigmatismus (erweitert):**
- ✅ z-Amplitude (Intensitätsabfall-Skala)
- ✅ z-Max (Clipping-Bereich)
- ✅ z0 (charakteristische Skala)
- ✅ Astigmatismus-Koeffizienten Ax, Ay (vorher: hardcoded)

### 🖥️ Desktop-App

**Build-System:**
- ✅ PyInstaller-Integration
- ✅ Cross-Platform Build-Scripts (Windows .bat + Mac/Linux .sh)
- ✅ Launcher mit Auto-Dependency-Check (`START_SIMULATOR.py`)
- ✅ Spec-File für optimierte Builds

**Features:**
- ✅ Standalone Executable (~200 MB)
- ✅ Keine Python-Installation nötig
- ✅ Portable (USB-Stick)
- ✅ Kein Konsolen-Fenster (GUI-only)

### 📚 Dokumentation

**Neue Dateien:**
- ✅ `ANLEITUNG_DESKTOP_APP.md` - Umfassende Desktop-App Anleitung
- ✅ `CHANGELOG.md` - Versionshistorie
- ✅ `build_app.spec` - PyInstaller Konfiguration
- ✅ `build_desktop_app.sh` / `.bat` - Build-Scripts
- ✅ `START_SIMULATOR.py` - Smart Launcher

**Aktualisiert:**
- ✅ `requirements.txt` - PyInstaller hinzugefügt
- ✅ Code-Kommentare - Alle neuen Funktionen dokumentiert

### 🔧 Technische Details

**Neue Klassen:**
- `PSFGeneratorOptimized` - Vektorisierte PSF-Berechnung
- `BackgroundGeneratorOptimized` - Mit Caching
- `TIFFSimulatorOptimized` - Hauptklasse mit Progress-Callbacks
- `TIFFSimulatorGUI_V4` - Erweiterte GUI
- `ToolTip` - Hilfe-Tooltips für GUI

**Backward Compatibility:**
- ✅ Alte APIs funktionieren weiterhin
- ✅ V3.0 GUI läuft mit V4.0 Engine
- ✅ Aliase: `TIFFSimulator = TIFFSimulatorOptimized`

### 🐛 Bugfixes

- ✅ Float32 statt Float64 (schneller, weniger Speicher)
- ✅ Robustere NaN/Inf-Behandlung
- ✅ Thread-safe UI-Updates
- ✅ Bessere Exception-Handling

---

## Version 3.0 - Oktober 2025

### Features
- ✅ Grundlegende TIFF-Simulation
- ✅ TDI-G0 & Tetraspecs Presets
- ✅ Polymerisationszeit-Modell
- ✅ Astigmatismus-Support
- ✅ z-Stack Kalibrierung
- ✅ Batch-Modus mit Presets
- ✅ Metadata-Export (JSON, TXT, CSV)
- ✅ GUI mit Scrollbarem Interface
- ✅ Jupyter Notebook Tutorial

### Physik
- ✅ Point Spread Function (2D Gaußsch)
- ✅ Brownsche Bewegung (normale/sub/confined Diffusion)
- ✅ Zeitabhängiger Diffusionskoeffizient D(t)
- ✅ Poisson-Noise + Read Noise
- ✅ Background mit Gradient
- ✅ Einfaches Blinking & Bleaching

### Performance
- ⚠️ Frame-für-Frame Verarbeitung (langsam bei großen TIFFs)
- ⚠️ Volle Bild-Meshgrids pro Spot
- ⚠️ Keine Parallelisierung

---

## Version 2.0 - Nicht veröffentlicht

Interne Entwicklungsversion

---

## Version 1.0 - Initial Release

Proof-of-Concept für Masterthesis
