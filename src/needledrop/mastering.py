from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.table import Table
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import sosfilt

ACTIVE_THRESHOLD_DB = -40.0
DEFAULT_TARGET_LUFS = -14.0
DEFAULT_CEILING_DBTP = -1.0
DC_OFFSET_WARN_DBFS = -60.0

# ITU-R BS.1770-4 pre-filter, in the bilinear form from Brecht De Man's loudness.py
# (reproduces the standard's 48 kHz tables at any sample rate)
K_SHELF_GAIN_DB = 3.99984385397
K_SHELF_Q = 0.7071752369554193
K_SHELF_FC_HZ = 1681.9744509555319
K_SHELF_VB_EXPONENT = 0.499666774155
K_HIGHPASS_Q = 0.5003270373253953
K_HIGHPASS_FC_HZ = 38.13547087613982

BURST_FRAME_MS = 5.0
BURST_HOP_SECONDS = BURST_FRAME_MS / 1000
BURST_ROLLING_FRAMES = 400
BURST_CREST_THRESHOLD_DB = 18.0
BURST_PEAK_FLOOR_MARGIN_DB = 2.0
BURST_MERGE_GAP_MS = 100.0
BURST_INFLUENCE_MS = 60.0

BALANCE_WINDOW_SECONDS = 61
BALANCE_DEAD_ZONE_DB = 0.8
BALANCE_RELEASE_DB = 0.4
BALANCE_CLAMP_DB = 3.0
BALANCE_SMOOTH_SECONDS = 5
BALANCE_BLOCK_SHAPE_STEP_DB = 0.3
BALANCE_BLOCK_SHAPE_MAX_FRACTION = 0.05

LIMITER_CEILING_MARGIN_DB = 0.1
LIMITER_OVERSAMPLE_FACTOR = 4
RESAMPLE_OPTIONS = ":filter_size=128:cutoff=0.98"
LIMITER_ATTACK_MS = 5
LIMITER_RELEASE_MS = 50
LIMITER_ENGAGED_THRESHOLD_DB = -0.1
LIMITER_TOUCHED_THRESHOLD_DB = -0.01
LIMITER_WARN_MAX_CUT_DB = 6.0
LIMITER_WARN_PCT = 2.0

NULL_TEST_WARN_DB = -60.0
NULL_TEST_FAIL_DB = -50.0

VERIFY_LUFS_TOLERANCE_LU = 0.5
VERIFY_PEAK_TOLERANCE_DB = 0.1

REPORT_BURST_ROW_CAP = 40
FFMPEG_ERROR_TAIL_LINES = 5


def _to_db(x) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-9))


@dataclass
class Diagnostics:
    samplerate: int
    num_seconds: int
    peak_dbfs: float
    clipped_samples: int
    dc_offset: tuple[float, float]
    per_second_peak: np.ndarray
    k_rms_l_db: np.ndarray
    k_rms_r_db: np.ndarray
    lr_correlation: np.ndarray
    per_minute_lr_diff_db: np.ndarray
    frame_peak: np.ndarray
    frame_peak_l: np.ndarray
    frame_peak_r: np.ndarray
    frame_mean_square: np.ndarray


@dataclass
class BurstEvent:
    timestamp_s: float
    duration_ms: float
    peak_dbfs: float
    crest_db: float
    louder_channel: str
    lr_peak_diff_db: float


@dataclass
class LoudnessMeasurement:
    integrated_lufs: float
    lra: float
    true_peak_dbfs: float


@dataclass
class BalanceCurve:
    diff_db: np.ndarray
    corr_db: np.ndarray
    block_shape_changes: int
    block_shape_ok: bool


@dataclass
class LimiterAccounting:
    hop_seconds: float
    total_seconds: float
    limited_seconds: float
    at_bursts_seconds: float
    on_music_seconds: float
    distinct_seconds_on_music: int
    pct_on_music: float
    max_cut_db: float
    max_cut_on_music_db: float
    warn: bool
    suggested_target: float | None


@dataclass
class NullTestResult:
    frames_checked: int
    pct_above_warn_db: float
    worst_residual_db: float
    passed: bool


@dataclass
class Snapshot:
    diag: Diagnostics
    lufs: float
    true_peak: float
    lra: float


def compute_frame_arrays(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hop = int(BURST_FRAME_MS * sr / 1000)
    n = len(audio) // hop
    if n == 0:
        empty = np.zeros(0)
        return empty, empty, empty, empty

    trimmed = audio[: n * hop].reshape(n, hop, 2)
    abs_frames = np.abs(trimmed)
    peak_l = abs_frames[:, :, 0].max(axis=1)
    peak_r = abs_frames[:, :, 1].max(axis=1)
    peak = np.maximum(peak_l, peak_r)
    mean_square = (trimmed.astype(np.float64) ** 2).mean(axis=(1, 2))
    return peak, peak_l, peak_r, mean_square


def k_weighting_sos(sr: int) -> np.ndarray:
    k = math.tan(math.pi * K_SHELF_FC_HZ / sr)
    vh = 10 ** (K_SHELF_GAIN_DB / 20)
    vb = vh**K_SHELF_VB_EXPONENT
    a0 = 1 + k / K_SHELF_Q + k * k
    shelf = [
        (vh + vb * k / K_SHELF_Q + k * k) / a0,
        2 * (k * k - vh) / a0,
        (vh - vb * k / K_SHELF_Q + k * k) / a0,
        1.0,
        2 * (k * k - 1) / a0,
        (1 - k / K_SHELF_Q + k * k) / a0,
    ]
    k = math.tan(math.pi * K_HIGHPASS_FC_HZ / sr)
    a0 = 1 + k / K_HIGHPASS_Q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / K_HIGHPASS_Q + k * k) / a0]
    return np.array([shelf, highpass])


def full_scale_threshold(subtype: str) -> float:
    match = re.search(r"(\d+)$", subtype)
    bits = int(match.group(1)) if match else 0
    if bits not in (8, 16, 24, 32):
        return 1.0
    return 1.0 - 2.0 ** (1 - bits)


def compute_diagnostics(path: Path, chunk_seconds: float = 1.0) -> Diagnostics:
    info = sf.info(path)
    if info.channels != 2:
        raise ValueError(f"{path.name} has {info.channels} channel(s); needledrop needs a stereo recording")
    sr = info.samplerate
    clip_threshold = full_scale_threshold(info.subtype)
    win = int(sr * chunk_seconds)
    hop = int(BURST_FRAME_MS * sr / 1000)
    sos = k_weighting_sos(sr)
    zi = np.zeros((sos.shape[0], 2, 2))

    peaks: list[float] = []
    k_rms_l: list[float] = []
    k_rms_r: list[float] = []
    correlation: list[float] = []
    dc: list[np.ndarray] = []
    clipped = 0

    frame_peak_chunks: list[np.ndarray] = []
    frame_peak_l_chunks: list[np.ndarray] = []
    frame_peak_r_chunks: list[np.ndarray] = []
    frame_mean_square_chunks: list[np.ndarray] = []
    leftover = np.empty((0, 2), dtype=np.float32)

    with sf.SoundFile(path) as f:
        while True:
            chunk = f.read(win, dtype="float32")
            if len(chunk) == 0:
                break
            abs_chunk = np.abs(chunk)
            peaks.append(float(abs_chunk.max()))
            clipped += int((abs_chunk >= clip_threshold).sum())

            chunk64 = chunk.astype(np.float64)
            dc.append(chunk64.mean(axis=0))
            weighted, zi = sosfilt(sos, chunk64, axis=0, zi=zi)
            power = (weighted**2).mean(axis=0)
            k_rms_l.append(float(np.sqrt(power[0])))
            k_rms_r.append(float(np.sqrt(power[1])))
            cross = float((weighted[:, 0] * weighted[:, 1]).mean())
            denominator = float(np.sqrt(power[0] * power[1]))
            correlation.append(cross / denominator if denominator > 0 else 0.0)

            buf = np.concatenate([leftover, chunk], axis=0) if len(leftover) else chunk
            peak, peak_l, peak_r, mean_square = compute_frame_arrays(buf, sr)
            leftover = buf[len(peak) * hop :]
            if len(peak):
                frame_peak_chunks.append(peak)
                frame_peak_l_chunks.append(peak_l)
                frame_peak_r_chunks.append(peak_r)
                frame_mean_square_chunks.append(mean_square)

    peaks_arr = np.array(peaks)
    k_rms_l_db = _to_db(np.array(k_rms_l))
    k_rms_r_db = _to_db(np.array(k_rms_r))
    diff = k_rms_l_db - k_rms_r_db
    active = (k_rms_l_db > ACTIVE_THRESHOLD_DB) & (k_rms_r_db > ACTIVE_THRESHOLD_DB)

    n_minutes = len(diff) // 60
    per_minute = np.full(n_minutes, np.nan)
    for i in range(n_minutes):
        window_active = active[i * 60 : (i + 1) * 60]
        window_diff = diff[i * 60 : (i + 1) * 60]
        if window_active.any():
            per_minute[i] = np.median(window_diff[window_active])

    dc_offset = tuple(np.mean(dc, axis=0).tolist()) if dc else (0.0, 0.0)

    return Diagnostics(
        samplerate=sr,
        num_seconds=len(peaks_arr),
        peak_dbfs=float(_to_db(peaks_arr.max())) if len(peaks_arr) else float("-inf"),
        clipped_samples=clipped,
        dc_offset=dc_offset,
        per_second_peak=peaks_arr,
        k_rms_l_db=k_rms_l_db,
        k_rms_r_db=k_rms_r_db,
        lr_correlation=np.array(correlation),
        per_minute_lr_diff_db=per_minute,
        frame_peak=np.concatenate(frame_peak_chunks) if frame_peak_chunks else np.zeros(0),
        frame_peak_l=np.concatenate(frame_peak_l_chunks) if frame_peak_l_chunks else np.zeros(0),
        frame_peak_r=np.concatenate(frame_peak_r_chunks) if frame_peak_r_chunks else np.zeros(0),
        frame_mean_square=np.concatenate(frame_mean_square_chunks) if frame_mean_square_chunks else np.zeros(0),
    )


def _active_seconds(diag: Diagnostics) -> np.ndarray:
    return (diag.k_rms_l_db > ACTIVE_THRESHOLD_DB) & (diag.k_rms_r_db > ACTIVE_THRESHOLD_DB)


def diagnostics_warnings(diag: Diagnostics) -> list[str]:
    warnings = []
    active = _active_seconds(diag)
    if active.any():
        median_corr = float(np.median(diag.lr_correlation[active]))
        if median_corr < 0:
            warnings.append(
                f"median L/R correlation {median_corr:+.2f}: one channel is probably polarity-inverted "
                "(check cartridge and phono wiring); not corrected"
            )
    dc_db = _to_db(np.abs(np.array(diag.dc_offset)))
    if dc_db.max() > DC_OFFSET_WARN_DBFS:
        warnings.append(
            f"DC offset {dc_db[0]:.0f} / {dc_db[1]:.0f} dBFS exceeds {DC_OFFSET_WARN_DBFS:.0f} dBFS; not removed"
        )
    return warnings


def detect_bursts(
    frame_peak: np.ndarray,
    frame_peak_l: np.ndarray,
    frame_peak_r: np.ndarray,
    frame_mean_square: np.ndarray,
    sr: int,
    per_second_peak: np.ndarray,
) -> list[BurstEvent]:
    hop = int(BURST_FRAME_MS * sr / 1000)
    n = len(frame_peak)
    if n == 0:
        return []

    pk = frame_peak
    rms = np.sqrt(uniform_filter1d(frame_mean_square, BURST_ROLLING_FRAMES, mode="nearest"))
    crest = _to_db(pk) - _to_db(rms)

    floor_db = _to_db(np.median(per_second_peak)) - BURST_PEAK_FLOOR_MARGIN_DB
    mask = (crest > BURST_CREST_THRESHOLD_DB) & (_to_db(pk) > floor_db)
    idx = np.where(mask)[0]

    merge_frames = int(BURST_MERGE_GAP_MS / BURST_FRAME_MS)
    groups: list[list[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= merge_frames:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])

    events = []
    for g in groups:
        t = g[0] * hop / sr
        dur_ms = (g[-1] - g[0] + 1) * hop * 1000 / sr
        peak_l = float(frame_peak_l[g].max())
        peak_r = float(frame_peak_r[g].max())
        louder = "L" if peak_l > peak_r else "R"
        events.append(
            BurstEvent(
                timestamp_s=t,
                duration_ms=dur_ms,
                peak_dbfs=float(_to_db(pk[g].max())),
                crest_db=float(crest[g].max()),
                louder_channel=louder,
                lr_peak_diff_db=float(_to_db(peak_l) - _to_db(peak_r)),
            )
        )
    return events


def _hysteresis_mask(values: np.ndarray, engage_above: float, release_below: float) -> np.ndarray:
    engaged = np.zeros(len(values), dtype=bool)
    on = False
    for i, value in enumerate(values):
        on = value >= release_below if on else value > engage_above
        engaged[i] = on
    return engaged


def balance_curve(level_l_db: np.ndarray, level_r_db: np.ndarray) -> BalanceCurve:
    n = len(level_l_db)
    diff = level_l_db - level_r_db
    active = (level_l_db > ACTIVE_THRESHOLD_DB) & (level_r_db > ACTIVE_THRESHOLD_DB)

    idx = np.arange(n)
    diff_interp = np.interp(idx, idx[active], diff[active]) if active.any() else np.zeros(n)

    med = median_filter(diff_interp, size=BALANCE_WINDOW_SECONDS, mode="reflect")
    engaged = _hysteresis_mask(np.abs(med), BALANCE_DEAD_ZONE_DB, BALANCE_RELEASE_DB)
    corr = np.where(engaged, med, 0.0)
    corr = np.clip(corr, -BALANCE_CLAMP_DB, BALANCE_CLAMP_DB)
    corr = uniform_filter1d(corr, BALANCE_SMOOTH_SECONDS, mode="nearest")

    changes = int((np.abs(np.diff(corr)) > BALANCE_BLOCK_SHAPE_STEP_DB).sum())
    threshold = int(BALANCE_BLOCK_SHAPE_MAX_FRACTION * n)
    block_shape_ok = changes <= threshold

    return BalanceCurve(
        diff_db=diff_interp,
        corr_db=corr,
        block_shape_changes=changes,
        block_shape_ok=block_shape_ok,
    )


def apply_balance(input_path: Path, output_path: Path, corr_db: np.ndarray, chunk_seconds: float = 1.0) -> None:
    with sf.SoundFile(input_path) as f:
        sr = f.samplerate
        win = int(sr * chunk_seconds)
        n = len(corr_db)
        centers = (np.arange(n) + 0.5) * sr
        gain_l_sec = 10 ** (-corr_db / 2 / 20)
        gain_r_sec = 10 ** (corr_db / 2 / 20)

        with sf.SoundFile(output_path, mode="w", samplerate=sr, channels=f.channels, subtype="PCM_24") as out:
            start = 0
            while True:
                chunk = f.read(win, dtype="float32")
                if len(chunk) == 0:
                    break
                t = np.arange(start, start + len(chunk))
                gain_l = np.interp(t, centers, gain_l_sec)
                gain_r = np.interp(t, centers, gain_r_sec)

                chunk64 = chunk.astype(np.float64)
                result = np.empty_like(chunk64)
                result[:, 0] = chunk64[:, 0] * gain_l
                result[:, 1] = chunk64[:, 1] * gain_r
                out.write(result)
                start += len(chunk)


def parse_ebur128_summary(stderr_text: str) -> LoudnessMeasurement:
    i_match = re.search(r"^\s*I:\s*(-?[\d.]+)\s*LUFS", stderr_text, re.MULTILINE)
    lra_match = re.search(r"^\s*LRA:\s*(-?[\d.]+)\s*LU\b", stderr_text, re.MULTILINE)
    peak_match = re.search(r"^\s*Peak:\s*(-?[\d.]+)\s*dBFS", stderr_text, re.MULTILINE)
    if not (i_match and lra_match and peak_match):
        raise RuntimeError("could not parse ebur128 summary from ffmpeg output")

    return LoudnessMeasurement(
        integrated_lufs=float(i_match.group(1)),
        lra=float(lra_match.group(1)),
        true_peak_dbfs=float(peak_match.group(1)),
    )


def compute_gain_db(target: float, measured_lufs: float) -> float:
    return target - measured_lufs


def limiter_linear_ceiling(ceiling_db: float) -> float:
    # alimiter clamps sample peaks; true-peak reconstruction can still overshoot
    # by a fraction of a dB, so the limit sits below the target ceiling
    return 10 ** ((ceiling_db - LIMITER_CEILING_MARGIN_DB) / 20)


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to run needledrop")


def _ffmpeg_error_tail(stderr_text: str) -> str:
    tail = "\n".join(line for line in (stderr_text or "").splitlines() if line.strip())
    return "\n".join(tail.splitlines()[-FFMPEG_ERROR_TAIL_LINES:])


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    ensure_ffmpeg_available()
    try:
        return subprocess.run(["ffmpeg", *args], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}):\n{_ffmpeg_error_tail(e.stderr)}") from e


def measure_loudness(path: Path) -> LoudnessMeasurement:
    result = _run_ffmpeg(
        ["-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"]
    )
    return parse_ebur128_summary(result.stderr)


def _render_filter(gain_db: float, sample_rate: int, limiter_ceiling_lin: float | None) -> str:
    oversampled = LIMITER_OVERSAMPLE_FACTOR * sample_rate
    stages = [f"volume={gain_db}dB", f"aresample={oversampled}{RESAMPLE_OPTIONS}"]
    if limiter_ceiling_lin is not None:
        stages.append(
            f"alimiter=limit={limiter_ceiling_lin:.4f}:attack={LIMITER_ATTACK_MS}:"
            f"release={LIMITER_RELEASE_MS}:level=false:latency=true"
        )
    stages.append(f"aresample={sample_rate}{RESAMPLE_OPTIONS}")
    return ",".join(stages)


def _render_audio(
    path: Path, output_path: Path, gain_db: float, sample_rate: int, limiter_ceiling_lin: float | None
) -> None:
    af = _render_filter(gain_db, sample_rate, limiter_ceiling_lin)
    _run_ffmpeg(["-y", "-nostats", "-i", str(path), "-af", af, "-c:a", "pcm_s24le", str(output_path)])


def render_limited(path: Path, output_path: Path, gain_db: float, ceiling_db: float, sample_rate: int) -> None:
    _render_audio(path, output_path, gain_db, sample_rate, limiter_linear_ceiling(ceiling_db))


def _read_exact(stream, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        piece = stream.read(remaining)
        if not piece:
            break
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def stream_reference_diff(
    stage_path: Path,
    rendered_path: Path,
    gain_db: float,
    sample_rate: int,
    chunk_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensure_ffmpeg_available()
    af = _render_filter(gain_db, sample_rate, None)
    win = int(sample_rate * chunk_seconds)
    hop = int(BURST_FRAME_MS * sample_rate / 1000)
    bytes_per_frame_sample = 2 * 4  # stereo float32

    peak_ref_chunks: list[np.ndarray] = []
    peak_out_chunks: list[np.ndarray] = []
    diff_peak_chunks: list[np.ndarray] = []
    leftover_ref = np.empty((0, 2), dtype=np.float32)
    leftover_out = np.empty((0, 2), dtype=np.float32)

    with tempfile.TemporaryFile() as stderr_file:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-nostats",
                "-i",
                str(stage_path),
                "-af",
                af,
                "-f",
                "f32le",
                "-c:a",
                "pcm_f32le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )

        try:
            with sf.SoundFile(rendered_path) as rendered:
                while True:
                    raw = _read_exact(proc.stdout, win * bytes_per_frame_sample)
                    usable = (len(raw) // bytes_per_frame_sample) * bytes_per_frame_sample
                    ref_chunk = np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, 2)
                    out_chunk = rendered.read(win, dtype="float32")

                    m = min(len(ref_chunk), len(out_chunk))
                    if m == 0:
                        break
                    ref_chunk = ref_chunk[:m]
                    out_chunk = out_chunk[:m]

                    buf_ref = np.concatenate([leftover_ref, ref_chunk]) if len(leftover_ref) else ref_chunk
                    buf_out = np.concatenate([leftover_out, out_chunk]) if len(leftover_out) else out_chunk
                    nb = min(len(buf_ref), len(buf_out))
                    buf_ref, buf_out = buf_ref[:nb], buf_out[:nb]

                    n_frames = nb // hop
                    used = n_frames * hop
                    leftover_ref, leftover_out = buf_ref[used:], buf_out[used:]

                    if n_frames:
                        r = np.abs(buf_ref[:used]).reshape(n_frames, hop, 2)
                        o = np.abs(buf_out[:used]).reshape(n_frames, hop, 2)
                        d = np.abs(buf_out[:used] - buf_ref[:used]).reshape(n_frames, hop, 2)
                        peak_ref_chunks.append(r.max(axis=(1, 2)))
                        peak_out_chunks.append(o.max(axis=(1, 2)))
                        diff_peak_chunks.append(d.max(axis=(1, 2)))
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            returncode = proc.wait()

        if returncode != 0:
            stderr_file.seek(0)
            text = stderr_file.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed (exit {returncode}):\n{_ffmpeg_error_tail(text)}")

    peak_ref = np.concatenate(peak_ref_chunks) if peak_ref_chunks else np.zeros(0)
    peak_out = np.concatenate(peak_out_chunks) if peak_out_chunks else np.zeros(0)
    diff_peak = np.concatenate(diff_peak_chunks) if diff_peak_chunks else np.zeros(0)
    return peak_ref, peak_out, diff_peak


def verify_loudness_target(integrated_lufs: float, true_peak_dbfs: float, target: float, ceiling: float) -> list[str]:
    violations = []
    if abs(integrated_lufs - target) > VERIFY_LUFS_TOLERANCE_LU:
        violations.append(
            f"integrated loudness {integrated_lufs:.2f} LUFS is more than "
            f"{VERIFY_LUFS_TOLERANCE_LU} LU off target {target:.2f}"
        )
    if true_peak_dbfs > ceiling + VERIFY_PEAK_TOLERANCE_DB:
        violations.append(
            f"true peak {true_peak_dbfs:.2f} dBTP exceeds ceiling {ceiling:.2f} + {VERIFY_PEAK_TOLERANCE_DB} dB"
        )
    return violations


def frame_peaks(audio: np.ndarray, hop: int) -> np.ndarray:
    n = len(audio) // hop
    channels = audio.shape[1] if audio.ndim == 2 else 1
    trimmed = audio[: n * hop]
    if audio.ndim == 2:
        return np.abs(trimmed).reshape(n, hop, channels).max(axis=(1, 2))
    return np.abs(trimmed).reshape(n, hop).max(axis=1)


def gain_reduction_db(peak_ref: np.ndarray, peak_out: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return 20 * np.log10(np.maximum(peak_out, eps) / np.maximum(peak_ref, eps))


def limiter_touched_mask(
    peak_ref: np.ndarray, peak_out: np.ndarray, threshold_db: float = LIMITER_TOUCHED_THRESHOLD_DB
) -> np.ndarray:
    return gain_reduction_db(peak_ref, peak_out) < threshold_db


def dilate_mask(mask: np.ndarray, radius_frames: int) -> np.ndarray:
    out = mask.copy()
    for k in range(1, radius_frames + 1):
        out |= np.roll(mask, k) | np.roll(mask, -k)
    return out


def burst_influence_mask(
    bursts: list[BurstEvent], num_frames: int, hop: int, sr: int, influence_ms: float = BURST_INFLUENCE_MS
) -> np.ndarray:
    mask = np.zeros(num_frames, dtype=bool)
    margin = influence_ms / 1000
    for b in bursts:
        start_s = b.timestamp_s - margin
        end_s = b.timestamp_s + b.duration_ms / 1000 + margin
        start = max(int(start_s * sr / hop), 0)
        end = min(int(end_s * sr / hop) + 1, num_frames)
        if start < end:
            mask[start:end] = True
    return mask


def limiter_accounting(
    peak_ref: np.ndarray,
    peak_out: np.ndarray,
    ceiling_db: float,
    bursts: list[BurstEvent],
    sr: int,
    hop_seconds: float,
    target: float,
) -> LimiterAccounting:
    n = min(len(peak_ref), len(peak_out))
    peak_ref = peak_ref[:n]
    peak_out = peak_out[:n]

    gr_db = gain_reduction_db(peak_ref, peak_out)
    ceiling_lin = limiter_linear_ceiling(ceiling_db)
    limited = (peak_ref > ceiling_lin) & (gr_db < LIMITER_ENGAGED_THRESHOLD_DB)

    hop = round(hop_seconds * sr)
    near = burst_influence_mask(bursts, n, hop, sr)
    on_music = limited & ~near
    at_bursts = limited & near

    total_seconds = n * hop_seconds
    limited_seconds = float(limited.sum()) * hop_seconds
    at_bursts_seconds = float(at_bursts.sum()) * hop_seconds
    on_music_seconds = float(on_music.sum()) * hop_seconds
    distinct_seconds_on_music = len({int(i * hop // sr) for i in np.where(on_music)[0]})
    pct_on_music = (on_music_seconds / total_seconds * 100) if total_seconds else 0.0

    max_cut_db = float(gr_db[limited].min()) if limited.any() else 0.0
    max_cut_on_music_db = float(gr_db[on_music].min()) if on_music.any() else 0.0

    warn = abs(max_cut_on_music_db) > LIMITER_WARN_MAX_CUT_DB or pct_on_music > LIMITER_WARN_PCT
    suggested_target = None
    if warn:
        if abs(max_cut_on_music_db) > LIMITER_WARN_MAX_CUT_DB:
            reduction = math.ceil(abs(max_cut_on_music_db) - LIMITER_WARN_MAX_CUT_DB)
        else:
            reduction = 1
        suggested_target = target - reduction

    return LimiterAccounting(
        hop_seconds=hop_seconds,
        total_seconds=total_seconds,
        limited_seconds=limited_seconds,
        at_bursts_seconds=at_bursts_seconds,
        on_music_seconds=on_music_seconds,
        distinct_seconds_on_music=distinct_seconds_on_music,
        pct_on_music=pct_on_music,
        max_cut_db=max_cut_db,
        max_cut_on_music_db=max_cut_on_music_db,
        warn=warn,
        suggested_target=suggested_target,
    )


def null_test(
    diff_peak: np.ndarray,
    out_peak: np.ndarray,
    exclude_mask: np.ndarray,
    warn_threshold_db: float = NULL_TEST_WARN_DB,
    fail_threshold_db: float = NULL_TEST_FAIL_DB,
) -> NullTestResult:
    n = min(len(diff_peak), len(out_peak))
    if n == 0:
        return NullTestResult(frames_checked=0, pct_above_warn_db=0.0, worst_residual_db=float("-inf"), passed=True)

    diff_peak = diff_peak[:n]
    out_peak = out_peak[:n]

    residual_db = 20 * np.log10(np.maximum(diff_peak, 1e-12) / np.maximum(out_peak, 1e-4))
    included = residual_db[~exclude_mask[:n]]

    if included.size == 0:
        return NullTestResult(frames_checked=0, pct_above_warn_db=0.0, worst_residual_db=float("-inf"), passed=True)

    worst = float(included.max())
    pct_above = float((included > warn_threshold_db).mean() * 100)
    return NullTestResult(
        frames_checked=int(included.size),
        pct_above_warn_db=pct_above,
        worst_residual_db=worst,
        passed=worst <= fail_threshold_db,
    )


def parse_timecode(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"invalid time code: {value}")


def trim_fade(
    audio: np.ndarray, sr: int, start: float | None, end: float | None, fade_in: float, fade_out: float
) -> np.ndarray:
    start_idx = max(0, min(int((start or 0.0) * sr), len(audio)))
    end_idx = len(audio) if end is None else max(start_idx, min(int(end * sr), len(audio)))
    out = audio[start_idx:end_idx].copy()

    fade_in_samples = min(int(fade_in * sr), len(out))
    if fade_in_samples > 0:
        ramp = np.linspace(0, 1, fade_in_samples)
        if out.ndim == 2:
            out[:fade_in_samples] *= ramp[:, np.newaxis]
        else:
            out[:fade_in_samples] *= ramp

    fade_out_samples = min(int(fade_out * sr), len(out))
    if fade_out_samples > 0:
        ramp = np.linspace(1, 0, fade_out_samples)
        if out.ndim == 2:
            out[-fade_out_samples:] *= ramp[:, np.newaxis]
        else:
            out[-fade_out_samples:] *= ramp

    return out


def apply_tpdf_dither(audio: np.ndarray, bit_depth: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    # libsndfile writes PCM without dithering; TPDF noise at 1 LSB avoids
    # quantization distortion on low-level material
    rng = rng or np.random.default_rng()
    lsb = 1.0 / (2 ** (bit_depth - 1))
    noise = (rng.uniform(-0.5, 0.5, audio.shape) + rng.uniform(-0.5, 0.5, audio.shape)) * lsb
    return audio + noise


def validate_bit_depth(bit_depth: int) -> None:
    if bit_depth not in (16, 24):
        raise ValueError(f"--bit-depth must be 16 or 24, got {bit_depth}")


def _fade_gain(pos: int, n: int, length: int, fade_in_samples: int, fade_out_samples: int) -> np.ndarray:
    idx = np.arange(pos, pos + n)
    gain = np.ones(n)

    if fade_in_samples > 0:
        mask = idx < fade_in_samples
        if fade_in_samples == 1:
            gain[mask] = 0.0
        else:
            gain[mask] *= idx[mask] / (fade_in_samples - 1)

    if fade_out_samples > 0:
        out_start = length - fade_out_samples
        mask = idx >= out_start
        if fade_out_samples == 1:
            gain[mask] *= 1.0
        else:
            gain[mask] *= 1.0 - (idx[mask] - out_start) / (fade_out_samples - 1)

    return gain


def finalize_output(
    rendered_path: Path,
    output_path: Path,
    sr: int,
    start: float | None,
    end: float | None,
    fade_in: float,
    fade_out: float,
    bit_depth: int,
    chunk_seconds: float = 1.0,
    move_when_possible: bool = True,
    rng: np.random.Generator | None = None,
) -> None:
    validate_bit_depth(bit_depth)
    no_trim_or_fade = start is None and end is None and fade_in == 0.0 and fade_out == 0.0
    if no_trim_or_fade and bit_depth == 24 and move_when_possible:
        os.replace(rendered_path, output_path)
        return

    rng = rng or np.random.default_rng()
    subtype = "PCM_16" if bit_depth == 16 else "PCM_24"

    with sf.SoundFile(rendered_path) as f_in:
        total = f_in.frames
        start_idx = max(0, min(int((start or 0.0) * sr), total))
        end_idx = total if end is None else max(start_idx, min(int(end * sr), total))
        length = end_idx - start_idx
        fade_in_samples = min(int(fade_in * sr), length)
        fade_out_samples = min(int(fade_out * sr), length)

        f_in.seek(start_idx)
        win = int(sr * chunk_seconds)

        with sf.SoundFile(output_path, mode="w", samplerate=sr, channels=f_in.channels, subtype=subtype) as f_out:
            pos = 0
            remaining = length
            while remaining > 0:
                chunk = f_in.read(min(win, remaining), dtype="float64")
                if len(chunk) == 0:
                    break
                gain = _fade_gain(pos, len(chunk), length, fade_in_samples, fade_out_samples)
                chunk = chunk * gain[:, np.newaxis]
                if bit_depth == 16:
                    chunk = apply_tpdf_dither(chunk, 16, rng)
                f_out.write(chunk)
                pos += len(chunk)
                remaining -= len(chunk)


def _diagnostics_rows(snap: Snapshot) -> list[tuple[str, str]]:
    lr = snap.diag.per_minute_lr_diff_db
    if lr.size == 0 or np.all(np.isnan(lr)):
        lr_range = "n/a"
    else:
        lr_range = f"{np.nanmin(lr):+.1f} / {np.nanmax(lr):+.1f} dB"

    active = _active_seconds(snap.diag)
    correlation = f"{np.median(snap.diag.lr_correlation[active]):+.2f}" if active.any() else "n/a"
    dc_db = _to_db(np.abs(np.array(snap.diag.dc_offset)))

    return [
        ("Peak", f"{snap.diag.peak_dbfs:.1f} dBFS"),
        ("True peak", f"{snap.true_peak:.1f} dBTP"),
        ("Integrated loudness", f"{snap.lufs:.1f} LUFS"),
        ("Loudness range", f"{snap.lra:.1f} LU"),
        ("Clipped samples", str(snap.diag.clipped_samples)),
        ("DC offset L/R", f"{dc_db[0]:.0f} / {dc_db[1]:.0f} dBFS"),
        ("Median L/R correlation", correlation),
        ("Per-minute L-R (K-weighted) min/max", lr_range),
    ]


def build_diagnostics_table(title: str, snap: Snapshot) -> Table:
    table = Table(title=title)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for label, value in _diagnostics_rows(snap):
        table.add_row(label, value)
    return table


def build_before_after_table(before: Snapshot, after: Snapshot) -> Table:
    table = Table(title="Before / After")
    table.add_column("Metric", style="bold")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    for (label, before_value), (_, after_value) in zip(_diagnostics_rows(before), _diagnostics_rows(after)):
        table.add_row(label, before_value, after_value)
    return table


def build_burst_table(bursts: list[BurstEvent], max_rows: int = REPORT_BURST_ROW_CAP) -> tuple[Table, int]:
    table = Table(title="Impulse bursts")
    table.add_column("Time", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Peak", justify="right")
    table.add_column("Crest", justify="right")
    table.add_column("Channel")
    table.add_column("L/R diff", justify="right")

    shown = bursts[:max_rows]
    for b in shown:
        minutes, seconds = divmod(b.timestamp_s, 60)
        table.add_row(
            f"{int(minutes)}:{seconds:05.2f}",
            f"{b.duration_ms:.0f} ms",
            f"{b.peak_dbfs:.1f} dBFS",
            f"{b.crest_db:.0f} dB",
            b.louder_channel,
            f"{b.lr_peak_diff_db:+.1f} dB",
        )
    return table, max(0, len(bursts) - max_rows)


def build_balance_table(curve: BalanceCurve) -> Table:
    table = Table(title="Balance correction")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Block-shape changes", str(curve.block_shape_changes))
    table.add_row("Block-shaped", "yes" if curve.block_shape_ok else "no")
    table.add_row("Correction min/max", f"{curve.corr_db.min():+.2f} / {curve.corr_db.max():+.2f} dB")
    table.add_row("Seconds corrected", str(int((curve.corr_db != 0).sum())))
    return table


def build_limiter_table(acc: LimiterAccounting) -> Table:
    table = Table(title="Limiter accounting")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total limited", f"{acc.limited_seconds:.2f} s")
    table.add_row("At detected bursts", f"{acc.at_bursts_seconds:.2f} s")
    table.add_row("On music", f"{acc.on_music_seconds:.2f} s")
    table.add_row("Distinct seconds touched (music)", str(acc.distinct_seconds_on_music))
    table.add_row("% of set limited (music)", f"{acc.pct_on_music:.2f}%")
    table.add_row("Max cut", f"{acc.max_cut_db:.1f} dB")
    table.add_row("Max cut on music", f"{acc.max_cut_on_music_db:.1f} dB")
    if acc.warn:
        suggestion = f" -> try --target {acc.suggested_target:.1f}" if acc.suggested_target is not None else ""
        table.add_row("Warning", f"[yellow]limiter working hard{suggestion}[/yellow]")
    return table


def format_null_test_line(result: NullTestResult) -> str:
    status = "[green]pass[/green]" if result.passed else "[red]FAIL[/red]"
    return (
        f"Null test: {result.pct_above_warn_db:.4f}% of {result.frames_checked} checked frames "
        f"differ by more than {NULL_TEST_WARN_DB:.0f} dB (worst residual {result.worst_residual_db:.1f} dB) [{status}]"
    )
