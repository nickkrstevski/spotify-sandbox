import os
import sys
import glob
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import pygame
import librosa


RECORDINGS_DIR = os.path.join(os.getcwd(), "recordings")
TARGET_SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64


def list_audio_files(directory: str) -> List[str]:
    files = []
    for ext in ("*.mp3", "*.wav"):
        files.extend(glob.glob(os.path.join(directory, ext)))
    files.sort()
    return files


def human_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_analysis(path: str) -> Tuple[np.ndarray, np.ndarray]:
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # shape: [n_mels, n_frames]
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )
    return S_db, times


def smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def load_beat_components(path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Precompute envelopes for kick, snare, hihat, vocals, bass.

    Returns a dict of arrays (one per frame) and the frame times.
    """
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    # Harmonic/percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)

    # Mel specs for total, percussive, and harmonic
    S_total = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    S_perc = librosa.feature.melspectrogram(
        y=y_perc, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    S_harm = librosa.feature.melspectrogram(
        y=y_harm, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    times = librosa.frames_to_time(
        np.arange(S_total.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )

    # Build mel frequency centers to define bands
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=sr / 2)

    def band_mask(low: float, high: float) -> np.ndarray:
        return (mel_freqs >= low) & (mel_freqs < high)

    bass_mask = band_mask(20, 150)
    mid_mask = band_mask(150, 2500)
    high_mask = band_mask(5000, sr / 2)
    vocal_mask = band_mask(300, 3400)

    # Energy per band over time
    bass_energy = S_total[bass_mask, :].sum(axis=0)
    mid_perc_energy = S_perc[mid_mask, :].sum(axis=0)
    high_perc_energy = S_perc[high_mask, :].sum(axis=0)
    vocal_harm_energy = S_harm[vocal_mask, :].sum(axis=0)

    # Onset-like envelopes via positive temporal differences
    def onset_envelope(x: np.ndarray, smooth_win: int = 7) -> np.ndarray:
        dx = np.diff(x, prepend=x[:1])
        dx = np.maximum(dx, 0)
        dx = smooth(dx, smooth_win)
        mx = np.maximum(dx.max(), 1e-9)
        return dx / mx

    kick_env = onset_envelope(bass_energy)
    snare_env = onset_envelope(mid_perc_energy)
    hihat_env = onset_envelope(high_perc_energy)

    # Normalize continuous levels
    def norm(x: np.ndarray) -> np.ndarray:
        mx = np.maximum(x.max(), 1e-9)
        return x / mx

    bass_level = norm(smooth(bass_energy, 9))
    vocal_level = norm(smooth(vocal_harm_energy, 9))

    components = {
        "kick": kick_env,
        "snare": snare_env,
        "hihat": hihat_env,
        "bass": bass_level,
        "vocals": vocal_level,
    }
    return components, times


def draw_spectrum(screen, spectrum_col: np.ndarray, W: int, H: int):
    # spectrum_col: shape [n_mels], values in dB (negative). Normalize to 0..1
    min_db, max_db = -60.0, 0.0
    vals = np.clip((spectrum_col - min_db) / (max_db - min_db), 0.0, 1.0)
    n = len(vals)
    margin = 40
    bar_w = (W - 2 * margin) / n
    max_bar_h = H - 2 * margin

    for i, v in enumerate(vals):
        h = v * max_bar_h
        x = margin + i * bar_w
        y = H - margin - h
        # Color gradient: low blue -> mid purple -> high pink
        color = (int(255 * v), int(50 + 150 * v), int(255 * (0.3 + 0.7 * v)))
        pygame.draw.rect(
            screen, color, pygame.Rect(int(x), int(y), max(1, int(bar_w * 0.9)), int(h))
        )


def draw_beats(screen, comps: Dict[str, np.ndarray], frame_idx: int, W: int, H: int):
    # Read envelopes
    def val(name: str) -> float:
        arr = comps[name]
        if frame_idx < 0:
            return 0.0
        if frame_idx >= len(arr):
            return float(arr[-1])
        return float(arr[frame_idx])

    kick = val("kick")
    snare = val("snare")
    hihat = val("hihat")
    bass = val("bass")
    vocals = val("vocals")

    # Layout
    cx_kick, cy = W * 0.25, H * 0.7
    cx_snare = W * 0.5
    cx_hat = W * 0.75
    max_r = min(W, H) * 0.12

    # Draw circles with intensity-based radius and glow
    def circle(color, cx, cy, v):
        r = max(4, v * max_r)
        col = (
            int(color[0] * (0.5 + 0.5 * v)),
            int(color[1] * (0.5 + 0.5 * v)),
            int(color[2] * (0.5 + 0.5 * v)),
        )
        pygame.draw.circle(screen, col, (int(cx), int(cy)), int(r), width=0)

    circle((255, 80, 60), cx_kick, cy, kick)  # kick: warm red
    circle((255, 200, 60), cx_snare, cy, snare)  # snare: yellow
    circle((120, 220, 255), cx_hat, cy, hihat)  # hihat: cyan

    # Draw horizontal bars for bass/vocals
    margin = 40
    bar_w = W - 2 * margin
    bass_h = int((H * 0.18) * bass)
    voc_h = int((H * 0.18) * vocals)
    pygame.draw.rect(
        screen, (100, 255, 120), pygame.Rect(margin, H * 0.15 - bass_h, bar_w, bass_h)
    )
    pygame.draw.rect(
        screen, (255, 120, 180), pygame.Rect(margin, H * 0.35 - voc_h, bar_w, voc_h)
    )
    font = pygame.font.SysFont("Arial", 16)
    screen.blit(font.render("Bass", True, (180, 255, 200)), (margin, int(H * 0.15) + 8))
    screen.blit(
        font.render("Vocals", True, (255, 200, 220)), (margin, int(H * 0.35) + 8)
    )


def draw_title(screen, title: str, W: int):
    font = pygame.font.SysFont("Arial", 24)
    surf = font.render(title, True, (230, 230, 230))
    screen.blit(surf, (20, 12))


def main():
    files = list_audio_files(RECORDINGS_DIR)
    if not files:
        print(f"No audio files found in {RECORDINGS_DIR}")
        sys.exit(1)

    pygame.init()
    pygame.mixer.init(frequency=TARGET_SR)
    W, H = 1100, 600
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Music Visualizer")
    clock = pygame.time.Clock()

    idx = 0
    paused = False
    mode = "bars"  # or "beats"
    elapsed = 0.0

    def play(index: int):
        path = files[index]
        title = human_name(path)
        print(f"Loading: {title}")
        # Precompute analysis for visualization
        spec, times = load_analysis(path)
        comps: Optional[Dict[str, np.ndarray]] = None
        comp_times: Optional[np.ndarray] = None

        def worker():
            nonlocal comps, comp_times
            comps, comp_times = load_beat_components(path)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        # Start playback
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        start_t = time.time()
        return title, spec, times, lambda: (comps, comp_times), start_t

    title, spec, times, comps_fn, start_t = play(idx)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key in (pygame.K_RIGHT, pygame.K_n):
                    idx = (idx + 1) % len(files)
                    title, spec, times, comps_fn, start_t = play(idx)
                    paused = False
                if event.key in (pygame.K_LEFT, pygame.K_p):
                    idx = (idx - 1) % len(files)
                    title, spec, times, comps_fn, start_t = play(idx)
                    paused = False
                if event.key in (pygame.K_SPACE,):
                    if paused:
                        pygame.mixer.music.unpause()
                        start_t = time.time() - elapsed
                        paused = False
                    else:
                        pygame.mixer.music.pause()
                        paused = True
                if event.key in (pygame.K_v,):
                    mode = "beats" if mode == "bars" else "bars"

        screen.fill((10, 10, 16))

        if not paused:
            elapsed = time.time() - start_t

        if mode == "bars":
            frame_idx = int(np.searchsorted(times, elapsed))
            frame_idx = max(0, min(frame_idx, spec.shape[1] - 1))
            spectrum_col = spec[:, frame_idx]
            draw_spectrum(screen, spectrum_col, W, H)
            draw_title(
                screen,
                f"{title}  [{idx + 1}/{len(files)}]  (SPACE=pause, ←/→ prev/next, V=mode)",
                W,
            )
        else:
            comps, comp_times = comps_fn()
            if comps is not None and comp_times is not None:
                frame_idx = int(np.searchsorted(comp_times, elapsed))
                frame_idx = max(0, min(frame_idx, len(comp_times) - 1))
                draw_beats(screen, comps, frame_idx, W, H)
                draw_title(
                    screen,
                    f"{title}  [{idx + 1}/{len(files)}]  (SPACE=pause, ←/→ prev/next, V=mode)  Beats",
                    W,
                )
            else:
                font = pygame.font.SysFont("Arial", 28)
                msg = font.render("Analyzing beat components...", True, (220, 220, 220))
                screen.blit(msg, (W // 2 - msg.get_width() // 2, H // 2 - 20))
                draw_title(
                    screen,
                    f"{title}  [{idx + 1}/{len(files)}]  (SPACE=pause, ←/→ prev/next, V=mode)",
                    W,
                )

        pygame.display.flip()
        clock.tick(60)

        # If music finished, auto-advance
        if not pygame.mixer.music.get_busy() and not paused:
            idx = (idx + 1) % len(files)
            title, spec, times, comps_fn, start_t = play(idx)


if __name__ == "__main__":
    main()
