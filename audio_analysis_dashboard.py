# audio_analysis_dashboard.py
"""
Streamlit dashboard for analyzing audio samples:
- Select dataset and sample indices
- For each sample: plot waveform, show global RMS, dB, and mel spectrogram

Run with:
    streamlit run audio_analysis_dashboard.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal

# --- Import custom audio processing functions ---
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent.parent))  # Ensure aug/ is in sys.path
from aug.utils import load_audio, pre_process_audio_mel
from aug.scripts.debug_noise_injection import compute_rms, compute_dbfs

# -------------------------------------------------------------------------
# Dataset Configuration (reuse/adapt from mel_spectrogram_viewer.py)
# -------------------------------------------------------------------------
class DatasetPaths(TypedDict):
    audio_dir: str

DATASET_CONFIG: Dict[str, DatasetPaths] = {
    "esc-50": {
        "audio_dir": "/projects/prjs1635/datasets/esc-50/groupped/constant",
    },
    "icbhi": {
        "audio_dir": "/projects/prjs1635/datasets(icbhi/ICBHI_final_database",
    },
    "synthetic": {
        "audio_dir": "/home/milerisoy/OPERA/datasets/icbhi/ICBHI_final_database",
    },
}

# -------------------------------------------------------------------------
# Data Handling
# -------------------------------------------------------------------------
class AudioDatasetLoader:
    """
    Loader for audio files by index, matching .wav/.flac/.mp3 files in a directory.
    """
    def __init__(self, dataset_name: str, root_dir: str = "", sr: int = 16000) -> None:
        self.dataset_name = dataset_name.lower()
        self.sr = sr
        if self.dataset_name not in DATASET_CONFIG:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Choose from {sorted(DATASET_CONFIG.keys())}."
            )
        config = DATASET_CONFIG[self.dataset_name]
        base_path = Path(root_dir)
        self.audio_dir = base_path / config["audio_dir"]

        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: '{self.audio_dir}'")

        # Gather all audio files (sorted for index-based access)
        self._file_paths: List[Path] = sorted(
            [p for ext in ("*.wav", "*.flac", "*.mp3") for p in self.audio_dir.glob(ext)]
        )
        if not self._file_paths:
            raise RuntimeError(f"No audio files found in '{self.audio_dir}'.")

    def __len__(self) -> int:
        return len(self._file_paths)

    def load(self, idx: int) -> tuple:
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} is out of bounds. Valid range is 0 to {len(self) - 1}.")
        audio_path = self._file_paths[idx]
        y = load_audio(str(audio_path), sr=self.sr)
        return y, self.sr, audio_path.name, audio_path

# -------------------------------------------------------------------------
# Streamlit UI Helpers
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading audio loader...")
def get_loader(name: str) -> AudioDatasetLoader:
    return AudioDatasetLoader(name)

def parse_indices(text: str) -> List[int]:
    """Convert a string like '0, 3 9' into a list of unique ints."""
    if not text:
        return []
    indices: set[int] = set()
    raw_parts = text.replace(",", " ").split()
    for part in raw_parts:
        try:
            indices.add(int(part))
        except ValueError:
            continue
    return sorted(list(indices))

def plot_waveform(y: np.ndarray, sr: int) -> None:
    plt.style.use('default')  # Fix for matplotlib/librosa style bug
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(np.arange(len(y)) / sr, y)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

def plot_mel_spectrogram(y: np.ndarray, sr: int) -> None:
    plt.style.use('default')  # Fix for matplotlib/librosa style bug
    mel_db = pre_process_audio_mel(y, f_max=sr//2, sr=sr, n_mels=128)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    img = ax.imshow(mel_db, aspect='auto', origin='lower', cmap='magma', vmin=-80, vmax=0)
    ax.set_title("Mel Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_ylim(0, 75)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

def a_weighting_filter(sr: int, n_fft: int) -> np.ndarray:
    """
    Returns the frequency response of the A-weighting filter for a given sample rate and FFT size.
    """
    # Frequencies for FFT bins
    f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    # A-weighting formula constants
    ra = (
        (12200 ** 2) * (f ** 4)
        / (
            (f ** 2 + 20.6 ** 2)
            * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
            * (f ** 2 + 12200 ** 2)
        )
    )
    a = 2.0 + 20 * np.log10(ra)
    return a

def compute_dbA(y: np.ndarray, sr: int) -> float:
    """
    Compute the A-weighted decibel value of an audio signal.
    """
    # Compute FFT
    n_fft = 2048
    Y = np.fft.rfft(y, n=n_fft)
    mag = np.abs(Y)
    # Get A-weighting curve
    a_weight = a_weighting_filter(sr, n_fft)
    # Apply A-weighting (in dB)
    mag_db = 20 * np.log10(mag + 1e-10)
    mag_dbA = mag_db + a_weight
    # Convert back to linear, compute RMS, then dB
    magA_lin = 10 ** (mag_dbA / 20)
    rmsA = np.sqrt(np.mean(magA_lin ** 2))
    dbA = 20 * np.log10(rmsA + 1e-10)
    return dbA

# -------------------------------------------------------------------------
# Main UI Application
# -------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Audio Analysis Dashboard")

    # --- Sidebar (Controls) ---
    st.sidebar.header("Configuration")
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset", options=sorted(DATASET_CONFIG.keys())
    )
    st.sidebar.subheader("Sample Selection")

    # Toggle for selection mode
    selection_mode = st.sidebar.toggle(
        "Select by index range (start,end) or specific indices",
        value=True,
        help="Toggle ON to select by index range (e.g., 10,20), OFF for specific indices (e.g., 0 5 10)"
    )
    if selection_mode:
        # Index range mode
        range_text = st.sidebar.text_input(
            "Index Range (start,end)",
            value="10,20",
            help="Provide start and end indices separated by a comma. E.g., '10,20' displays samples 10 to 20 inclusive."
        )
        try:
            start_str, end_str = range_text.split(",")
            start_idx = int(start_str.strip())
            end_idx = int(end_str.strip())
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            requested_indices = list(range(start_idx, end_idx + 1))
        except Exception:
            st.warning("Invalid range input. Please provide two integers separated by a comma, e.g., '10,20'.")
            requested_indices = []
    else:
        # Specific indices mode
        indices_text = st.sidebar.text_input(
            "Sample Indices",
            value="0 1 2",
            help="Provide indices separated by space or comma. E.g., '0 5 10' or '0,5,10'"
        )
        requested_indices = parse_indices(indices_text)
    display_btn = st.sidebar.button("Display", use_container_width=True, type="primary")

    if not display_btn:
        st.info("Configure your selection in the sidebar and click 'Display'.")
        st.stop()

    # --- Data Loading ---
    try:
        loader = get_loader(dataset_choice)
        num_available = len(loader)
        st.sidebar.success(f"Found {num_available} audio files for '{dataset_choice}'.\n\n"
                          f"Valid index range: 0 to {num_available - 1}.")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    # --- Index Parsing and Validation ---
    valid_indices = [i for i in requested_indices if 0 <= i < num_available]
    invalid_indices = [i for i in requested_indices if not (0 <= i < num_available)]

    if invalid_indices:
        st.warning(f"Indices out of range or invalid and were skipped: {invalid_indices[:10]}...")

    if not valid_indices:
        st.info("No valid samples to display based on your selection.")
        st.stop()

    # --- Main Panel (Display) ---
    st.header(f"Displaying {len(valid_indices)} Sample(s) for: {dataset_choice.upper()}")

    for idx in valid_indices:
        st.subheader(f"Sample Index: {idx}")
        try:
            y, sr, filename, audio_path = loader.load(idx)
            col1, col2 = st.columns(2)
            with col1:
                plot_waveform(y, sr)
                rms = compute_rms(y)
                db = compute_dbfs(y)
                dbA = compute_dbA(y, sr)
                st.metric("Global RMS", f"{rms:.4f}")
                st.metric("RMS (dB)", f"{db:.2f} dB")
                st.metric("A-weighted (dBA)", f"{dbA:.2f} dBA")
            with col2:
                plot_mel_spectrogram(y, sr)
            st.caption(f"File: {filename}")
            st.audio(str(audio_path), format="audio/wav")
        except Exception as e:
            st.error(f"Error loading index {idx}: {e}")

if __name__ == "__main__":
    main() 