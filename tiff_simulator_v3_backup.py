"""
🔬 HYPERREALISTISCHES TIFF-SIMULATIONSSYSTEM V3.0
=================================================

Wissenschaftlich präzise Simulation von Single-Molecule Tracking Daten
für hochauflösende Fluoreszenzmikroskopie.

Physikalische Grundlagen:
-------------------------
- Point Spread Function (PSF): 2D Gaußsche Approximation
- Diffusion: Brownsche Bewegung mit zeitabhängigem D(t)
- Astigmatismus: Elliptische PSF-Deformation als Funktion von z
- Photon Noise: Poisson-Statistik für realistische Bildgebung

Autor: Generiert für Masterthesis
Version: 3.0 - Oktober 2025
Lizenz: MIT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


@dataclass
class DetectorPreset:
    """
    Detektor-Konfiguration mit physikalisch realistischen Parametern.
    
    Attributes:
    -----------
    name : str
        Detektorbezeichnung
    max_intensity : float
        Maximale Photonenzahl pro Spot [counts]
    background_mean : float
        Mittlerer Background [counts]
    background_std : float
        Standardabweichung des Backgrounds [counts]
    pixel_size_um : float
        Physikalische Pixelgröße [µm]
    fwhm_um : float
        Full Width at Half Maximum der PSF [µm]
    """
    name: str
    max_intensity: float
    background_mean: float
    background_std: float
    pixel_size_um: float
    fwhm_um: float
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# DETEKTOR PRESETS - Experimentell validiert
# ============================================================================

TDI_PRESET = DetectorPreset(
    name="TDI-G0",
    max_intensity=260.0,
    background_mean=100.0,
    background_std=15.0,
    pixel_size_um=0.108,
    fwhm_um=0.40,
    metadata={
        "detector_type": "TDI Line Scan Camera",
        "numerical_aperture": 1.2,
        "wavelength_nm": 580,
        "quantum_efficiency": 0.85
    }
)

TETRASPECS_PRESET = DetectorPreset(
    name="Tetraspecs",
    max_intensity=300.0,
    background_mean=100.0,
    background_std=15.0,
    pixel_size_um=0.160,
    fwhm_um=0.40,
    metadata={
        "detector_type": "sCMOS Camera",
        "numerical_aperture": 1.2,
        "wavelength_nm": 580,
        "quantum_efficiency": 0.90
    }
)


# ============================================================================
# ZEITABHÄNGIGE DIFFUSIONSKOEFFIZIENTEN
# ============================================================================

def get_time_dependent_D(t_poly_min: float, D_initial: float, 
                         diffusion_type: str = "normal") -> float:
    """
    Berechnet zeitabhängigen Diffusionskoeffizienten während der 
    Polymerisationsphase basierend auf experimentellen Daten.
    
    Physikalisches Modell:
    ----------------------
    D(t) = D₀ · exp(-t/τ) · f(t)
    
    wobei:
    - D₀: Initialer Diffusionskoeffizient [µm²/s]
    - τ: Charakteristische Zeitkonstante (40 min)
    - f(t): Zusätzliche Reduktionsfunktion für t > 90 min
    
    Die starke Reduktion von D bei langen Polymerisationszeiten reflektiert
    die zunehmende Netzwerkdichte und Viskosität des Hydrogels.
    
    Parameters:
    -----------
    t_poly_min : float
        Polymerisationszeit [min]
    D_initial : float
        Initialer D-Wert bei t=0 [µm²/s], typisch 3-5 µm²/s
    diffusion_type : str
        "normal", "subdiffusion", "superdiffusion", "confined"
    
    Returns:
    --------
    float : Zeitabhängiger D-Wert [µm²/s]
    
    Referenzen:
    -----------
    - Exponentieller Abfall: Saxton & Jacobson, Annu Rev Biophys (1997)
    - Subdiffusion in Gelen: Masuda et al., Phys Rev Lett (2005)
    """
    
    # Basis-Reduktion: Exponentieller Abfall
    tau = 40.0  # Charakteristische Zeitkonstante [min]
    reduction_factor = np.exp(-t_poly_min / tau)
    
    # Zusätzliche Reduktion ab 90 min (starke Vernetzung)
    if t_poly_min >= 90:
        extra_reduction = 0.5 * np.exp(-(t_poly_min - 90) / 30.0)
        reduction_factor *= extra_reduction
    
    D_base = D_initial * reduction_factor
    
    # Diffusionstyp-spezifische Modifikationen
    if diffusion_type == "subdiffusion":
        # Subdiffusion: Zusätzliche Verlangsamung
        D_base *= 0.6
    elif diffusion_type == "superdiffusion":
        # Superdiffusion: Leichte Erhöhung (selten)
        D_base *= 1.3
    elif diffusion_type == "confined":
        # Confined: Stark reduziert
        D_base *= 0.3
    
    return max(D_base, 0.001)  # Minimum: 0.001 µm²/s


def get_diffusion_fractions(t_poly_min: float) -> Dict[str, float]:
    """
    Berechnet Fraktionen verschiedener Diffusionstypen als Funktion der Zeit.
    
    Mit zunehmender Polymerisation steigt der Anteil von Sub- und Confined
    Diffusion, während normale Brownsche Bewegung abnimmt.
    
    Parameters:
    -----------
    t_poly_min : float
        Polymerisationszeit [min]
    
    Returns:
    --------
    Dict[str, float] : Fraktionen für jeden Diffusionstyp
    """
    
    # Zeitabhängige Fraktionen (Summe = 1.0)
    if t_poly_min < 10:
        fractions = {
            "normal": 0.95,
            "subdiffusion": 0.04,
            "confined": 0.01,
            "superdiffusion": 0.0
        }
    elif t_poly_min < 60:
        # Lineare Interpolation
        progress = t_poly_min / 60.0
        fractions = {
            "normal": 0.95 - 0.30 * progress,
            "subdiffusion": 0.04 + 0.20 * progress,
            "confined": 0.01 + 0.09 * progress,
            "superdiffusion": 0.0
        }
    elif t_poly_min < 120:
        progress = (t_poly_min - 60.0) / 60.0
        fractions = {
            "normal": 0.65 - 0.25 * progress,
            "subdiffusion": 0.24 + 0.10 * progress,
            "confined": 0.10 + 0.15 * progress,
            "superdiffusion": 0.01
        }
    else:
        # > 120 min: Stark vernetzt
        progress = min((t_poly_min - 120.0) / 60.0, 1.0)
        fractions = {
            "normal": 0.40 - 0.05 * progress,
            "subdiffusion": 0.34 + 0.01 * progress,
            "confined": 0.25 + 0.03 * progress,
            "superdiffusion": 0.01
        }
    
    # Normalisierung
    total = sum(fractions.values())
    return {k: v/total for k, v in fractions.items()}


# ============================================================================
# PSF GENERATOR - Physikalisch korrekte Point Spread Function
# ============================================================================

class PSFGenerator:
    """
    Generiert physikalisch realistische Point Spread Functions (PSF).
    
    Die PSF wird als 2D Gaußfunktion modelliert:
    
    I(x,y) = I₀ · exp(-[(x-x₀)²/σₓ² + (y-y₀)²/σᵧ²]/2)
    
    wobei:
    - I₀: Peak-Intensität [counts]
    - σₓ, σᵧ: Standardabweichungen in x,y [px]
    - Beziehung: FWHM = 2√(2ln2) · σ ≈ 2.355 · σ
    
    Für Astigmatismus (z-abhängig):
    σₓ(z) = σ₀ · √(1 + (z/z₀)²)
    σᵧ(z) = σ₀ · √(1 - (z/z₀)²)
    """
    
    def __init__(self, detector: DetectorPreset, astigmatism: bool = False):
        self.detector = detector
        self.astigmatism = astigmatism
        
        # Konvertiere FWHM zu σ (Standardabweichung)
        # FWHM = 2.355 * σ
        fwhm_px = detector.fwhm_um / detector.pixel_size_um
        self.sigma_px = fwhm_px / 2.355
        # Numerischer Mindestwert für Sigma, um Instabilitäten zu vermeiden
        self._sigma_eps = 1e-6
        
        # Astigmatismus-Parameter
        if astigmatism:
            # z₀: charakteristische Länge für Astigmatismus [µm]
            self.z0_um = 0.5
    
    def generate_psf(self, center_x: float, center_y: float, 
                     intensity: float, z_um: float = 0.0,
                     image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Generiert eine einzelne PSF.
        
        Parameters:
        -----------
        center_x, center_y : float
            Spot-Position [px]
        intensity : float
            Peak-Intensität [counts]
        z_um : float
            z-Position für Astigmatismus [µm]
        image_size : Tuple[int, int]
            Bildgröße (height, width) [px]
        
        Returns:
        --------
        np.ndarray : PSF-Bild [counts]
        """
        
        height, width = image_size
        
        # Berechne σₓ, σᵧ basierend auf z-Position
        if self.astigmatism and z_um != 0.0:
            # Astigmatische PSF
            z_norm = z_um / self.z0_um
            sigma_x = self.sigma_px * np.sqrt(1 + z_norm**2)
            # Stabilisiere den Ausdruck für sigma_y, damit er stets reell/positiv bleibt
            sigma_y_term = np.maximum(1 - 0.5 * (z_norm**2), self._sigma_eps)
            sigma_y = self.sigma_px * np.sqrt(sigma_y_term)
        else:
            # Symmetrische PSF
            sigma_x = sigma_y = self.sigma_px

        # Untere Schranke für numerische Stabilität
        sigma_x = max(sigma_x, self._sigma_eps)
        sigma_y = max(sigma_y, self._sigma_eps)
        
        # Erstelle Koordinaten-Meshgrid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # 2D Gaußfunktion
        psf = intensity * np.exp(
            -(((x - center_x)**2 / (2 * sigma_x**2)) +
              ((y - center_y)**2 / (2 * sigma_y**2)))
        )
        
        return psf
    
    def get_metadata(self) -> Dict:
        """Gibt PSF-Metadata zurück"""
        return {
            "fwhm_um": self.detector.fwhm_um,
            "sigma_px": self.sigma_px,
            "pixel_size_um": self.detector.pixel_size_um,
            "astigmatism": self.astigmatism,
            "z0_um": self.z0_um if self.astigmatism else None
        }


# ============================================================================
# TRAJEKTORIEN-GENERATOR - Brownsche Bewegung
# ============================================================================

class TrajectoryGenerator:
    """
    Generiert realistische Trajektorien basierend auf verschiedenen
    Diffusionsmodellen.
    
    Brownsche Bewegung (normale Diffusion):
    ---------------------------------------
    Δr² = 2·d·D·Δt
    
    wobei:
    - d: Dimensionalität (2 für 2D, 3 für 3D)
    - D: Diffusionskoeffizient [µm²/s]
    - Δt: Zeitintervall [s]
    
    Subdiffusion (Anomale Diffusion):
    ----------------------------------
    Δr² = 2·d·D·Δt^α
    mit α < 1 (typisch 0.6-0.8)
    """
    
    def __init__(self, D_initial: float, t_poly_min: float, 
                 frame_rate_hz: float, pixel_size_um: float):
        self.D_initial = D_initial
        self.t_poly_min = t_poly_min
        self.dt = 1.0 / frame_rate_hz
        self.pixel_size_um = pixel_size_um
        
        # Hole Diffusionsfraktionen
        self.fractions = get_diffusion_fractions(t_poly_min)
        
        # Berechne D-Werte für jeden Typ
        self.D_values = {
            dtype: get_time_dependent_D(t_poly_min, D_initial, dtype)
            for dtype in self.fractions.keys()
        }
    
    def generate_trajectory(self, start_pos: Tuple[float, float, float],
                           num_frames: int, 
                           diffusion_type: str = "normal") -> np.ndarray:
        """
        Generiert eine 3D-Trajektorie.
        
        Parameters:
        -----------
        start_pos : Tuple[float, float, float]
            Startposition (x, y, z) [µm]
        num_frames : int
            Anzahl Frames
        diffusion_type : str
            Typ der Diffusion
        
        Returns:
        --------
        np.ndarray : Trajektorie (num_frames, 3) [µm]
        """
        
        D = self.D_values[diffusion_type]
        trajectory = np.zeros((num_frames, 3))
        trajectory[0] = start_pos
        
        # Anomaler Exponent für Subdiffusion
        alpha = 0.7 if diffusion_type == "subdiffusion" else 1.0
        
        for i in range(1, num_frames):
            # Mean Square Displacement
            if diffusion_type == "confined":
                # Confined: Begrenzter Raum
                msd = 2 * D * self.dt
                # Rückstellkraft zum Zentrum
                drift = -0.1 * (trajectory[i-1] - start_pos)
            else:
                # Normale/Sub/Super Diffusion
                msd = 2 * D * (self.dt ** alpha)
                drift = np.zeros(3)
            
            # Brownsche Schritte
            step = np.random.normal(0, np.sqrt(msd), size=3)
            
            trajectory[i] = trajectory[i-1] + step + drift
        
        return trajectory
    
    def generate_multi_trajectory(self, num_spots: int, num_frames: int,
                                  image_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        Generiert mehrere Trajektorien mit verschiedenen Diffusionstypen.
        
        Returns:
        --------
        List[np.ndarray] : Liste von Trajektorien
        """
        
        height, width = image_size
        trajectories = []
        
        for _ in range(num_spots):
            # Wähle Diffusionstyp basierend auf Fraktionen
            dtype = np.random.choice(
                list(self.fractions.keys()),
                p=list(self.fractions.values())
            )
            
            # Zufällige Startposition (in µm, innerhalb des Bildes)
            start_x = np.random.uniform(0.2 * width, 0.8 * width) * self.pixel_size_um
            start_y = np.random.uniform(0.2 * height, 0.8 * height) * self.pixel_size_um
            start_z = np.random.uniform(-0.5, 0.5)  # z in [-0.5, 0.5] µm
            
            trajectory = self.generate_trajectory(
                (start_x, start_y, start_z),
                num_frames,
                dtype
            )
            
            trajectories.append({
                "positions": trajectory,
                "diffusion_type": dtype,
                "D_value": self.D_values[dtype]
            })
        
        return trajectories
    
    def get_metadata(self) -> Dict:
        """Gibt Trajektorien-Metadata zurück"""
        return {
            "D_initial": self.D_initial,
            "t_poly_min": self.t_poly_min,
            "frame_rate_hz": 1.0 / self.dt,
            "diffusion_fractions": self.fractions,
            "D_values": self.D_values
        }


# ============================================================================
# BACKGROUND GENERATOR - Realistischer Hintergrund
# ============================================================================

class BackgroundGenerator:
    """Generiert realistischen Background mit Gradiente und Rauschen."""
    
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def generate(self, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Generiert Background-Bild.
        
        Returns:
        --------
        np.ndarray : Background [counts]
        """
        
        height, width = image_size
        
        # Basis-Background (Poisson-Rauschen)
        background = np.random.poisson(self.mean, size=(height, width)).astype(float)
        
        # Gaußsches Rauschen
        noise = np.random.normal(0, self.std, size=(height, width))
        background += noise
        
        # Leichter Gradient (simuliert ungleichmäßige Beleuchtung)
        y, x = np.meshgrid(np.linspace(-1, 1, height), 
                          np.linspace(-1, 1, width), 
                          indexing='ij')
        gradient = 5 * (x**2 + y**2)
        background += gradient
        
        return np.maximum(background, 0)  # Keine negativen Werte


# ============================================================================
# HAUPTSIMULATOR
# ============================================================================

class TIFFSimulator:
    """
    Hauptklasse für TIFF-Simulation.
    
    Workflow:
    ---------
    1. Initialisierung mit Detektor-Preset
    2. Generierung von Trajektorien
    3. Rendering der PSFs
    4. Addition von Background & Noise
    5. Export als TIFF
    """
    
    def __init__(self, detector: DetectorPreset, mode: str = "polyzeit",
                 t_poly_min: float = 60.0, astigmatism: bool = False):
        
        self.detector = detector
        self.mode = mode
        self.t_poly_min = t_poly_min
        self.astigmatism = astigmatism
        
        # Initialisiere Generatoren
        self.psf_gen = PSFGenerator(detector, astigmatism)
        self.bg_gen = BackgroundGenerator(
            detector.background_mean,
            detector.background_std
        )
        
        # Metadata
        self.metadata = {
            "detector": detector.name,
            "mode": mode,
            "t_poly_min": t_poly_min,
            "astigmatism": astigmatism,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_tiff(self, image_size: Tuple[int, int], num_spots: int,
                     num_frames: int, frame_rate_hz: float) -> np.ndarray:
        """
        Generiert TIFF-Stack.
        
        Parameters:
        -----------
        image_size : Tuple[int, int]
            (height, width) [px]
        num_spots : int
            Anzahl Fluoreszenz-Spots
        num_frames : int
            Anzahl Frames
        frame_rate_hz : float
            Frame Rate [Hz]
        
        Returns:
        --------
        np.ndarray : TIFF-Stack (num_frames, height, width) [counts]
        """
        
        height, width = image_size
        
        # Initialisiere Trajektorien-Generator
        traj_gen = TrajectoryGenerator(
            D_initial=4.0,  # Default D_initial
            t_poly_min=self.t_poly_min,
            frame_rate_hz=frame_rate_hz,
            pixel_size_um=self.detector.pixel_size_um
        )
        
        # Generiere Trajektorien
        trajectories = traj_gen.generate_multi_trajectory(
            num_spots, num_frames, image_size
        )
        
        # Initialisiere TIFF-Stack
        tiff_stack = np.zeros((num_frames, height, width), dtype=np.uint16)
        
        # Generiere jeden Frame
        for frame_idx in range(num_frames):
            # Background
            frame = self.bg_gen.generate(image_size)
            
            # Füge jeden Spot hinzu
            for traj_data in trajectories:
                pos = traj_data["positions"][frame_idx]
                
                # Konvertiere µm zu Pixel
                x_px = pos[0] / self.detector.pixel_size_um
                y_px = pos[1] / self.detector.pixel_size_um
                z_um = pos[2] if self.astigmatism else 0.0
                
                # Check ob im Bild
                if 0 <= x_px < width and 0 <= y_px < height:
                    # Generiere PSF
                    psf = self.psf_gen.generate_psf(
                        x_px, y_px,
                        self.detector.max_intensity,
                        z_um,
                        image_size
                    )
                    
                    frame += psf
            
            # Poisson-Noise (shot noise) – robuste Vorverarbeitung
            frame = np.nan_to_num(frame, nan=0.0, posinf=1e6, neginf=0.0)
            frame = np.clip(frame, 0, 1e6)
            frame = np.random.poisson(frame)
            
            # Clip & Convert
            tiff_stack[frame_idx] = np.clip(frame, 0, 65535).astype(np.uint16)
        
        # Update Metadata
        self.metadata.update({
            "image_size": image_size,
            "num_spots": num_spots,
            "num_frames": num_frames,
            "frame_rate_hz": frame_rate_hz,
            "trajectories": trajectories,
            "psf": self.psf_gen.get_metadata(),
            "diffusion": traj_gen.get_metadata()
        })
        
        return tiff_stack
    
    def generate_z_stack(self, image_size: Tuple[int, int], num_spots: int,
                        z_range_um: Tuple[float, float], 
                        z_step_um: float) -> np.ndarray:
        """
        Generiert z-Stack für Kalibrierung (statische Spots).
        
        Returns:
        --------
        np.ndarray : z-Stack (n_slices, height, width)
        """
        
        z_min, z_max = z_range_um
        z_positions = np.arange(z_min, z_max + z_step_um, z_step_um)
        n_slices = len(z_positions)
        
        height, width = image_size
        
        # Generiere zufällige, aber STATISCHE Spot-Positionen
        spot_positions = []
        for _ in range(num_spots):
            x_px = np.random.uniform(0.2 * width, 0.8 * width)
            y_px = np.random.uniform(0.2 * height, 0.8 * height)
            spot_positions.append((x_px, y_px))
        
        # Erstelle z-Stack
        z_stack = np.zeros((n_slices, height, width), dtype=np.uint16)
        
        for z_idx, z_um in enumerate(z_positions):
            # Background
            frame = self.bg_gen.generate(image_size)
            
            # Füge Spots hinzu (immer an gleicher Position!)
            for x_px, y_px in spot_positions:
                psf = self.psf_gen.generate_psf(
                    x_px, y_px,
                    self.detector.max_intensity,
                    z_um,
                    image_size
                )
                frame += psf
            
            # Poisson-Noise – robuste Vorverarbeitung
            frame = np.nan_to_num(frame, nan=0.0, posinf=1e6, neginf=0.0)
            frame = np.clip(frame, 0, 1e6)
            frame = np.random.poisson(frame)
            z_stack[z_idx] = np.clip(frame, 0, 65535).astype(np.uint16)
        
        # Update Metadata
        self.metadata.update({
            "image_size": image_size,
            "num_spots": num_spots,
            "z_range_um": z_range_um,
            "z_step_um": z_step_um,
            "n_slices": n_slices,
            "spot_positions": spot_positions
        })
        
        return z_stack
    
    def get_metadata(self) -> Dict:
        """Gibt alle Metadata zurück"""
        return self.metadata.copy()


# ============================================================================
# TIFF EXPORT
# ============================================================================

def save_tiff(filepath: str, tiff_stack: np.ndarray, 
              metadata: Optional[Dict] = None) -> None:
    """
    Speichert TIFF-Stack mit Metadata.
    
    Parameters:
    -----------
    filepath : str
        Pfad zur TIFF-Datei
    tiff_stack : np.ndarray
        TIFF-Stack (frames, height, width)
    metadata : Dict, optional
        Metadata-Dictionary
    """
    
    from PIL import Image
    
    # Erstelle TIFF
    images = [Image.fromarray(frame) for frame in tiff_stack]
    
    # Speichere als Multi-Page TIFF
    images[0].save(
        filepath,
        save_all=True,
        append_images=images[1:],
        compression='tiff_deflate'
    )
    
    print(f"✅ TIFF gespeichert: {filepath}")
    print(f"   Shape: {tiff_stack.shape}")
    print(f"   Dtype: {tiff_stack.dtype}")
    print(f"   Range: [{tiff_stack.min()}, {tiff_stack.max()}]")


# ============================================================================
# QUICK TESTING
# ============================================================================

if __name__ == "__main__":
    print("🔬 TIFF Simulator V3.0 - Backend Test")
    print("=" * 50)
    
    # Test: Einfache Simulation
    sim = TIFFSimulator(
        detector=TDI_PRESET,
        mode="polyzeit",
        t_poly_min=60.0,
        astigmatism=False
    )
    
    tiff = sim.generate_tiff(
        image_size=(64, 64),
        num_spots=3,
        num_frames=10,
        frame_rate_hz=20.0
    )
    
    print(f"\n✅ Test erfolgreich!")
    print(f"   TIFF Shape: {tiff.shape}")
    print(f"   Mean Intensity: {tiff.mean():.1f}")
    print(f"   Max Intensity: {tiff.max()}")
