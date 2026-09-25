"""Measure how much ffmpeg's swr resampler colours a 4x up/down round trip (the mastering limiter's
oversampling stage): passband flatness, round-trip residual and wall time per candidate option set.

Run: uv run python scripts/probe_resampler_roundtrip.py [--out-dir DIR] [--rates 44100,48000] [--chains B,D]
Results that chose RESAMPLE_OPTIONS in src/needledrop/mastering.py: .specs/mastering-best-practices-review.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

SECONDS = 20
TARGET_RMS = 10 ** (-20 / 20)
BL_CUTOFF_HZ = 20000
BL_NUMTAPS = 2001
MAX_LAG = 2000
TRIM_SEC = 0.5


def chains_for(sr: int) -> dict[str, str]:
    r4 = 4 * sr
    return {
        "A": f"aresample=192000,aresample={sr}",
        "B": f"aresample={r4},aresample={sr}",
        "C": f"aresample={r4}:filter_size=64,aresample={sr}:filter_size=64",
        "D": f"aresample={r4}:filter_size=128:cutoff=0.98,aresample={sr}:filter_size=128:cutoff=0.98",
        "E": f"aresample={r4}:filter_size=256:cutoff=0.98,aresample={sr}:filter_size=256:cutoff=0.98",
        "F": f"aresample={r4}:filter_size=512:cutoff=0.985,aresample={sr}:filter_size=512:cutoff=0.985",
        "G": (
            f"aresample={r4}:filter_size=256:cutoff=0.98:filter_type=blackman_nuttall,"
            f"aresample={sr}:filter_size=256:cutoff=0.98:filter_type=blackman_nuttall"
        ),
    }


def scale_to_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    return x * (target_rms / np.sqrt(np.mean(x**2)))


def make_white(sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = sr * SECONDS
    return np.stack([scale_to_rms(rng.standard_normal(n), TARGET_RMS) for _ in range(2)], axis=1).astype(np.float32)


def make_bl_noise(sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = sr * SECONDS
    taps = scipy.signal.firwin(BL_NUMTAPS, BL_CUTOFF_HZ, fs=sr)
    channels = [scale_to_rms(scipy.signal.filtfilt(taps, [1.0], rng.standard_normal(n)), TARGET_RMS) for _ in range(2)]
    return np.stack(channels, axis=1).astype(np.float32)


def run_ffmpeg(in_path: Path, out_path: Path, af: str) -> tuple[int, str, float]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(in_path), "-af", af, "-c:a", "pcm_f32le", str(out_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stderr, time.perf_counter() - t0


def find_lag(x: np.ndarray, y: np.ndarray, max_lag: int = MAX_LAG) -> int:
    corr = scipy.signal.correlate(y, x, mode="full", method="fft")
    zero_idx = len(x) - 1
    window = corr[zero_idx - max_lag : zero_idx + max_lag + 1]
    return int(np.argmax(np.abs(window))) - max_lag


def align_and_trim(x: np.ndarray, y: np.ndarray, lag: int, sr: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        x_a, y_a = x[: len(x) - lag] if lag > 0 else x, y[lag:]
    else:
        x_a, y_a = x[-lag:], y[: len(y) + lag]
    overlap = min(len(x_a), len(y_a))
    trim = round(TRIM_SEC * sr)
    return x_a[:overlap][trim:-trim], y_a[:overlap][trim:-trim]


def self_test_alignment() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(50000)
    for true_lag in (-137, 0, 483, 2000, -2000):
        if true_lag >= 0:
            y = np.concatenate([np.zeros(true_lag), x])[: len(x)]
        else:
            y = np.concatenate([x[-true_lag:], np.zeros(-true_lag)])[: len(x)]
        assert find_lag(x, y) == true_lag, f"alignment self-test failed at lag {true_lag}"


def db(ratio: float) -> float:
    return 20 * np.log10(ratio)


def residual_metrics(in_x: np.ndarray, out_y: np.ndarray, sr: int) -> tuple[float, float]:
    diff = out_y - in_x
    rms_in = np.sqrt(np.mean(in_x**2))
    taps = scipy.signal.firwin(501, 19000, fs=sr)
    diff_lp = scipy.signal.filtfilt(taps, [1.0], diff)
    return db(np.sqrt(np.mean(diff**2)) / rms_in), db(np.sqrt(np.mean(diff_lp**2)) / rms_in)


def mag_response(in_x: np.ndarray, out_y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, p_in = scipy.signal.welch(in_x, fs=sr, nperseg=8192)
    _, p_out = scipy.signal.welch(out_y, fs=sr, nperseg=8192)
    return freqs, 10 * np.log10(p_out / p_in)


def first_crossing(freqs: np.ndarray, ratio_db: np.ndarray, threshold: float) -> float | None:
    below = ratio_db < threshold
    idx = int(np.argmax(below))
    if not below[idx]:
        return None
    if idx == 0:
        return float(freqs[0])
    f0, f1, v0, v1 = freqs[idx - 1], freqs[idx], ratio_db[idx - 1], ratio_db[idx]
    return float(f1) if v1 == v0 else float(f0 + (threshold - v0) / (v1 - v0) * (f1 - f0))


def measure(in_path: Path, out_path: Path, sr: int, label: str) -> dict:
    x, _ = sf.read(in_path, dtype="float64", always_2d=True)
    y, _ = sf.read(out_path, dtype="float64", always_2d=True)
    lag = find_lag(x[:, 0], y[:, 0])
    xa, ya = align_and_trim(x[:, 0], y[:, 0], lag, sr)
    out = {"lag": lag, "len_diff": len(y) - len(x)}
    if label == "bl_noise":
        out["residual_db"], out["residual_lp19k_db"] = residual_metrics(xa, ya, sr)
    else:
        freqs, ratio_db = mag_response(xa, ya, sr)
        band = (freqs >= 20) & (freqs <= 18000)
        out["max_dev_20_18k_db"] = float(np.max(np.abs(ratio_db[band])))
        out["dev_20k_db"] = float(np.interp(20000, freqs, ratio_db))
        out["f_m0.1db"] = first_crossing(freqs, ratio_db, -0.1)
        out["f_m1db"] = first_crossing(freqs, ratio_db, -1.0)
        out["f_m3db"] = first_crossing(freqs, ratio_db, -3.0)
    return out


def fmt_hz(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=None, help="work dir for signals and renders (default: temp)")
    parser.add_argument("--rates", default="44100,48000", help="comma-separated sample rates")
    parser.add_argument("--chains", default="A,B,C,D,E,F,G", help="comma-separated chain letters")
    args = parser.parse_args()

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="resampler-probe-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = [int(r) for r in args.rates.split(",")]
    wanted = args.chains.split(",")

    self_test_alignment()
    results: dict[str, dict] = {}
    for sr in rates:
        signals = {"bl_noise": make_bl_noise(sr, 2000 + sr), "white": make_white(sr, 1000 + sr)}
        for label, data in signals.items():
            sf.write(out_dir / f"{label}_{sr}.wav", data, sr, subtype="FLOAT")

        print(f"\n{sr} Hz")
        header = f"{'chain':6}{'resid dB':>10}{'resid<19k':>11}{'|dev|<18k':>11}{'dev@20k':>9}{'-0.1dB Hz':>11}{'-1dB Hz':>9}{'-3dB Hz':>9}{'lag':>5}{'wall s':>8}"
        print(header)
        for chain, af in chains_for(sr).items():
            if chain not in wanted:
                continue
            entry: dict = {"af": af}
            for label in signals:
                in_path = out_dir / f"{label}_{sr}.wav"
                out_path = out_dir / f"out_{chain}_{label}_{sr}.wav"
                rc, err, wall = run_ffmpeg(in_path, out_path, af)
                if rc != 0:
                    entry[f"{label}_error"] = err.strip()[-500:]
                    continue
                entry[f"{label}_wall_s"] = wall
                entry.update({f"{label}_{k}": v for k, v in measure(in_path, out_path, sr, label).items()})
            results[f"{sr}|{chain}"] = entry
            if any(k.endswith("_error") for k in entry):
                print(f"{chain:6}FAILED: {entry.get('bl_noise_error') or entry.get('white_error')}")
                continue
            print(
                f"{chain:6}{entry['bl_noise_residual_db']:>10.2f}{entry['bl_noise_residual_lp19k_db']:>11.2f}"
                f"{entry['white_max_dev_20_18k_db']:>11.4f}{entry['white_dev_20k_db']:>9.2f}"
                f"{fmt_hz(entry['white_f_m0.1db']):>11}{fmt_hz(entry['white_f_m1db']):>9}{fmt_hz(entry['white_f_m3db']):>9}"
                f"{entry['bl_noise_lag']:>5}{(entry['bl_noise_wall_s'] + entry['white_wall_s']) / 2:>8.3f}"
            )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nrenders and results.json in {out_dir}")


if __name__ == "__main__":
    main()
