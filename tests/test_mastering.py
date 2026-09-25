import io
import shutil
import subprocess
import tracemalloc

import numpy as np
import pytest
import soundfile as sf

from needledrop import mastering as m

DB = lambda x: 20 * np.log10(np.maximum(x, 1e-9))


def sine_stereo(sr, seconds, left_gain_db, right_gain_db, freq=50.0, base_amp=0.2):
    n = seconds * sr
    t = np.arange(n) / sr
    carrier = np.sin(2 * np.pi * freq * t)
    left_gain = np.repeat(10 ** (np.asarray(left_gain_db) / 20), sr)
    right_gain = np.repeat(10 ** (np.asarray(right_gain_db) / 20), sr)
    left = carrier * base_amp * left_gain
    right = carrier * base_amp * right_gain
    return np.stack([left, right], axis=1).astype(np.float32)


def per_second_rms_db(audio, sr):
    n = len(audio) // sr
    rms = np.sqrt((audio[: n * sr] ** 2).reshape(n, sr, 2).mean(axis=1))
    return DB(rms[:, 0]), DB(rms[:, 1])


def apply_balance_reference(audio, sr, corr_db):
    n = len(corr_db)
    centers = (np.arange(n) + 0.5) * sr
    t = np.arange(len(audio))
    gain_l = np.interp(t, centers, 10 ** (-corr_db / 2 / 20))
    gain_r = np.interp(t, centers, 10 ** (corr_db / 2 / 20))

    audio64 = audio.astype(np.float64)
    out = np.empty_like(audio64)
    out[:, 0] = audio64[:, 0] * gain_l
    out[:, 1] = audio64[:, 1] * gain_r
    return out


# --- balance ---


def test_balance_curve_matches_a_block_imbalance_within_tolerance():
    sr, seconds = 1000, 600
    left_gain = np.zeros(seconds)
    left_gain[240:480] = 2.3
    audio = sine_stereo(sr, seconds, left_gain, np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)

    assert curve.corr_db[300:420].mean() == pytest.approx(2.3, abs=0.2)
    assert curve.block_shape_ok


def test_balance_curve_leaves_dead_zone_seconds_untouched():
    sr, seconds = 1000, 600
    left_gain = np.zeros(seconds)
    left_gain[240:480] = 2.3
    audio = sine_stereo(sr, seconds, left_gain, np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)

    assert np.abs(curve.corr_db[:150]).max() < 0.05
    assert np.abs(curve.corr_db[540:600]).max() < 0.05


def test_balance_curve_clamps_correction_to_max_db():
    sr, seconds = 1000, 600
    left_gain = np.zeros(seconds)
    left_gain[240:480] = 5.0
    audio = sine_stereo(sr, seconds, left_gain, np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)

    assert curve.corr_db.max() <= m.BALANCE_CLAMP_DB + 1e-6


def test_balance_curve_smoothing_bounds_the_per_second_step():
    sr, seconds = 1000, 600
    left_gain = np.zeros(seconds)
    left_gain[240:480] = 2.3
    audio = sine_stereo(sr, seconds, left_gain, np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)
    max_step = np.max(np.abs(np.diff(curve.corr_db)))

    assert max_step <= 2.3 / m.BALANCE_SMOOTH_SECONDS + 0.05


def test_balance_curve_flags_a_wandering_correction_as_not_block_shaped():
    sr, seconds = 1000, 600
    left_gain = np.zeros(seconds)
    for i in range(0, seconds, 20):
        left_gain[i : i + 20] = 2.5 if (i // 20) % 2 == 0 else -2.5
    audio = sine_stereo(sr, seconds, left_gain, np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)

    assert not curve.block_shape_ok
    assert curve.block_shape_changes > int(m.BALANCE_BLOCK_SHAPE_MAX_FRACTION * seconds)


# --- streaming diagnostics / apply_balance on a small real file ---


@pytest.fixture
def clicks_and_block_wav(tmp_path):
    sr = 48000
    seconds = 20
    t = np.arange(seconds * sr) / sr
    tone = 0.1 * np.sin(2 * np.pi * 200 * t)
    left, right = tone.copy(), tone.copy()

    left_gain = np.ones(seconds)
    left_gain[8:14] = 10 ** (2.3 / 20)
    left = left * np.repeat(left_gain, sr)

    click_times = [3.0, 16.0]
    click_len = int(0.005 * sr)
    for ct in click_times:
        start = int(ct * sr)
        left[start : start + click_len] = 1.0
        right[start : start + click_len] = 1.0

    audio = np.stack([left, right], axis=1).astype(np.float32)
    path = tmp_path / "clicks_and_block.wav"
    sf.write(path, audio, sr, subtype="FLOAT")
    return path, audio, sr, seconds, click_times


def test_streaming_diagnostics_produces_expected_frame_arrays_and_detects_clicks(clicks_and_block_wav):
    path, _, sr, seconds, click_times = clicks_and_block_wav
    diag = m.compute_diagnostics(path)

    hop = int(m.BURST_FRAME_MS * sr / 1000)
    expected_frames = seconds * sr // hop
    assert len(diag.frame_peak) == expected_frames
    assert len(diag.frame_peak_l) == expected_frames
    assert len(diag.frame_peak_r) == expected_frames
    assert len(diag.frame_mean_square) == expected_frames

    bursts = m.detect_bursts(
        diag.frame_peak, diag.frame_peak_l, diag.frame_peak_r, diag.frame_mean_square, sr, diag.per_second_peak
    )

    assert len(bursts) == len(click_times)
    got_times = sorted(b.timestamp_s for b in bursts)
    for got, expected_t in zip(got_times, sorted(click_times)):
        assert got == pytest.approx(expected_t, abs=0.01)


def test_streaming_apply_balance_matches_non_streaming_expectation(clicks_and_block_wav, tmp_path):
    path, audio, sr, seconds, _ = clicks_and_block_wav
    diag = m.compute_diagnostics(path)
    curve = m.balance_curve(diag.k_rms_l_db, diag.k_rms_r_db)

    expected = apply_balance_reference(audio, sr, curve.corr_db)

    balanced_path = tmp_path / "balanced.wav"
    m.apply_balance(path, balanced_path, curve.corr_db)
    balanced, _ = sf.read(balanced_path)

    assert len(balanced) == len(audio)

    n = seconds
    exp_rms = np.sqrt((expected[: n * sr] ** 2).reshape(n, sr, 2).mean(axis=1))
    got_rms = np.sqrt((balanced[: n * sr].astype(np.float64) ** 2).reshape(n, sr, 2).mean(axis=1))
    exp_diff = DB(exp_rms[:, 0]) - DB(exp_rms[:, 1])
    got_diff = DB(got_rms[:, 0]) - DB(got_rms[:, 1])
    assert np.abs(got_diff - exp_diff).max() < 0.05


def test_diagnostics_and_apply_balance_streaming_stays_under_memory_budget(tmp_path):
    sr = 48000
    seconds = 60
    n = seconds * sr
    rng = np.random.default_rng(2)
    audio = (0.3 * rng.uniform(-1, 1, (n, 2))).astype(np.float32)
    path = tmp_path / "mem_test.wav"
    sf.write(path, audio, sr, subtype="PCM_16")
    del audio

    diag = m.compute_diagnostics(path)
    curve = m.balance_curve(diag.k_rms_l_db, diag.k_rms_r_db)
    out_path = tmp_path / "mem_test_balanced.wav"

    tracemalloc.start()
    m.compute_diagnostics(path)
    m.apply_balance(path, out_path, curve.corr_db)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 64 * 1024 * 1024


# --- gain / loudnorm json ---


def test_compute_gain_db():
    assert m.compute_gain_db(target=-14.0, measured_lufs=-18.5) == pytest.approx(4.5)


def test_parse_ebur128_summary_extracts_integrated_lra_and_peak():
    stderr_text = (
        "[Parsed_ebur128_0 @ 0x1] t: 2.999977   TARGET:-23 LUFS    M: -21.1 S: -21.1     I: -21.1 LUFS       LRA:   0.0 LU\n"
        "[Parsed_ebur128_0 @ 0x1] Summary:\n"
        "\n"
        "  Integrated loudness:\n"
        "    I:         -14.1 LUFS\n"
        "    Threshold: -31.1 LUFS\n"
        "\n"
        "  Loudness range:\n"
        "    LRA:         6.2 LU\n"
        "    Threshold: -41.1 LUFS\n"
        "    LRA low:   -41.1 LUFS\n"
        "    LRA high:  -21.1 LUFS\n"
        "\n"
        "  True peak:\n"
        "    Peak:      -1.0 dBFS\n"
    )
    summary = m.parse_ebur128_summary(stderr_text)
    assert summary.integrated_lufs == pytest.approx(-14.1)
    assert summary.lra == pytest.approx(6.2)
    assert summary.true_peak_dbfs == pytest.approx(-1.0)


# --- ffmpeg subprocess wrapping (mocked) ---


def test_ensure_ffmpeg_available_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        m.ensure_ffmpeg_available()


def test_measure_loudness_runs_ebur128_in_true_peak_mode_and_parses_the_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    captured = {}
    summary = (
        "  Integrated loudness:\n"
        "    I:         -18.0 LUFS\n"
        "  Loudness range:\n"
        "    LRA:         5.0 LU\n"
        "  True peak:\n"
        "    Peak:      -2.0 dBFS\n"
    )

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="", stderr=summary)

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    measurement = m.measure_loudness(tmp_path / "in.wav")

    assert measurement.integrated_lufs == pytest.approx(-18.0)
    assert measurement.true_peak_dbfs == pytest.approx(-2.0)
    assert measurement.lra == pytest.approx(5.0)
    af = captured["args"][captured["args"].index("-af") + 1]
    assert af.startswith("ebur128=")
    assert "peak=true" in af


def test_render_limited_includes_alimiter_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    m.render_limited(tmp_path / "in.wav", tmp_path / "out.wav", gain_db=3.0, ceiling_db=-1.0, sample_rate=48000)

    af = captured["args"][captured["args"].index("-af") + 1]
    assert "alimiter" in af
    assert "volume=3.0dB" in af
    assert af.split(",")[-1].startswith("aresample=48000")


# --- streaming reference/rendered comparison (mocked ffmpeg pipe) ---


def test_stream_reference_diff_matches_direct_computation_with_partial_block_and_length_mismatch(
    monkeypatch, tmp_path
):
    sr = 1000
    hop = int(m.BURST_FRAME_MS * sr / 1000)
    rng = np.random.default_rng(3)
    ref_full = rng.uniform(-0.5, 0.5, (2050, 2)).astype(np.float32)
    out_full = ref_full.copy()
    out_full[100:110] += 0.02
    out_full_extended = np.concatenate([out_full, np.zeros((3, 2), dtype=np.float32)])

    rendered_path = tmp_path / "rendered.wav"
    sf.write(rendered_path, out_full_extended, sr, subtype="FLOAT")

    raw_bytes = ref_full.astype("<f4").tobytes()

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(raw_bytes)

        def wait(self):
            return 0

    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return FakeProcess()

    monkeypatch.setattr(m.subprocess, "Popen", fake_popen)

    peak_ref, peak_out, diff_peak = m.stream_reference_diff(
        tmp_path / "stage.wav", rendered_path, gain_db=0.0, sample_rate=sr, chunk_seconds=0.5
    )

    af = captured["args"][captured["args"].index("-af") + 1]
    assert "alimiter" not in af

    expected_n = 2050 // hop
    expected_ref = m.frame_peaks(ref_full[:2050], hop)
    expected_out = m.frame_peaks(out_full[:2050], hop)
    expected_diff = m.frame_peaks(out_full[:2050] - ref_full[:2050], hop)

    assert len(peak_ref) == expected_n
    np.testing.assert_allclose(peak_ref, expected_ref, atol=1e-6)
    np.testing.assert_allclose(peak_out, expected_out, atol=1e-6)
    np.testing.assert_allclose(diff_peak, expected_diff, atol=1e-6)


def test_stream_reference_diff_raises_with_stderr_tail_on_nonzero_exit(monkeypatch, tmp_path):
    sr = 1000
    rendered_path = tmp_path / "rendered.wav"
    sf.write(rendered_path, np.zeros((10, 2), dtype=np.float32), sr, subtype="FLOAT")

    stderr_text = "ffmpeg version 9.0.1\n" + "config line\n" * 50 + "stage.wav: No space left on device\n"

    class FakeProcess:
        def __init__(self, stderr_file):
            self.stdout = io.BytesIO(b"")
            stderr_file.write(stderr_text.encode())

        def wait(self):
            return 228

    def fake_popen(args, stdout=None, stderr=None, **kwargs):
        return FakeProcess(stderr)

    monkeypatch.setattr(m.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError) as excinfo:
        m.stream_reference_diff(tmp_path / "stage.wav", rendered_path, gain_db=0.0, sample_rate=sr)

    message = str(excinfo.value)
    assert "exit 228" in message
    assert "No space left on device" in message
    assert "ffmpeg version" not in message


# --- verify ---


def test_verify_loudness_target_passes_within_tolerance():
    assert m.verify_loudness_target(-14.2, -1.05, target=-14.0, ceiling=-1.0) == []


def test_verify_loudness_target_flags_off_target_loudness():
    violations = m.verify_loudness_target(-16.0, -1.05, target=-14.0, ceiling=-1.0)
    assert violations
    assert "loudness" in violations[0]


def test_verify_loudness_target_flags_true_peak_over_ceiling():
    violations = m.verify_loudness_target(-14.0, -0.5, target=-14.0, ceiling=-1.0)
    assert violations
    assert "peak" in violations[0]


# --- burst detection ---


def test_detect_bursts_finds_a_click_above_the_surrounding_tone():
    sr, seconds = 8000, 3.0
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t)
    left, right = tone.copy(), tone.copy()
    click_start = int(1.5 * sr)
    click_len = int(0.005 * sr)
    left[click_start : click_start + click_len] = 1.0
    audio = np.stack([left, right], axis=1).astype(np.float32)
    per_second_peak = np.array([0.1, 0.1, 0.1])

    frame_peak, frame_peak_l, frame_peak_r, frame_mean_square = m.compute_frame_arrays(audio, sr)
    events = m.detect_bursts(frame_peak, frame_peak_l, frame_peak_r, frame_mean_square, sr, per_second_peak)

    assert len(events) == 1
    assert events[0].timestamp_s == pytest.approx(1.5, abs=0.01)
    assert events[0].louder_channel == "L"
    assert events[0].crest_db > m.BURST_CREST_THRESHOLD_DB


def test_detect_bursts_finds_nothing_in_a_steady_tone():
    sr, seconds = 8000, 3.0
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t)
    audio = np.stack([tone, tone], axis=1).astype(np.float32)
    per_second_peak = np.array([0.1, 0.1, 0.1])

    frame_peak, frame_peak_l, frame_peak_r, frame_mean_square = m.compute_frame_arrays(audio, sr)
    assert m.detect_bursts(frame_peak, frame_peak_l, frame_peak_r, frame_mean_square, sr, per_second_peak) == []


# --- limiter accounting ---


def _flat_limiter_inputs(n_frames, limited_indices, cut_db, ceiling_lin=0.881):
    peak_ref = np.full(n_frames, 0.5)
    peak_out = peak_ref.copy()
    peak_ref[limited_indices] = ceiling_lin + 0.02
    peak_out[limited_indices] = peak_ref[limited_indices] * 10 ** (cut_db / 20)
    return peak_ref, peak_out


def test_limiter_accounting_uses_total_time_not_distinct_seconds_for_the_pct_warning():
    sr = 48000
    hop_seconds = m.BURST_HOP_SECONDS
    hop = round(hop_seconds * sr)
    seconds_total = 1000
    n_frames = seconds_total * (sr // hop)

    limited_indices = np.array([sec * (sr // hop) for sec in range(0, seconds_total, 2)])
    peak_ref, peak_out = _flat_limiter_inputs(n_frames, limited_indices, cut_db=-0.5)

    acc = m.limiter_accounting(peak_ref, peak_out, ceiling_db=-1.0, bursts=[], sr=sr, hop_seconds=hop_seconds, target=-14.0)

    assert acc.distinct_seconds_on_music == len(limited_indices)
    assert acc.pct_on_music < m.LIMITER_WARN_PCT
    assert not acc.warn
    assert acc.suggested_target is None


def test_limiter_accounting_warns_when_max_cut_exceeds_threshold():
    sr = 48000
    hop_seconds = m.BURST_HOP_SECONDS
    n_frames = 10
    peak_ref, peak_out = _flat_limiter_inputs(n_frames, [0], cut_db=-7.6)

    acc = m.limiter_accounting(peak_ref, peak_out, ceiling_db=-1.0, bursts=[], sr=sr, hop_seconds=hop_seconds, target=-14.0)

    assert acc.warn
    assert acc.max_cut_on_music_db == pytest.approx(-7.6, abs=0.01)
    assert acc.suggested_target == pytest.approx(-16.0)


def test_limiter_accounting_warns_when_total_limited_time_exceeds_pct_threshold():
    sr = 48000
    hop_seconds = m.BURST_HOP_SECONDS
    n_frames = 1000
    limited_indices = np.arange(30)
    peak_ref, peak_out = _flat_limiter_inputs(n_frames, limited_indices, cut_db=-0.5)

    acc = m.limiter_accounting(peak_ref, peak_out, ceiling_db=-1.0, bursts=[], sr=sr, hop_seconds=hop_seconds, target=-14.0)

    assert acc.pct_on_music > m.LIMITER_WARN_PCT
    assert acc.warn
    assert acc.suggested_target == pytest.approx(-15.0)


# --- null test ---


def test_null_test_passes_on_identical_arrays():
    rng = np.random.default_rng(0)
    ref = rng.uniform(-0.5, 0.5, (1000, 2)).astype(np.float32)
    out = ref.copy()
    exclude_mask = np.zeros(200, dtype=bool)
    hop = round(m.BURST_HOP_SECONDS * 1000)
    diff_peak = m.frame_peaks(out - ref, hop)
    out_peak = m.frame_peaks(out, hop)

    result = m.null_test(diff_peak, out_peak, exclude_mask)

    assert result.passed
    assert result.worst_residual_db < -100
    assert result.pct_above_warn_db == 0.0


def test_null_test_fails_on_a_perturbation_outside_the_excluded_mask():
    rng = np.random.default_rng(0)
    ref = rng.uniform(-0.5, 0.5, (1000, 2)).astype(np.float32)
    out = ref.copy()
    out[500:520] += 0.3
    exclude_mask = np.zeros(200, dtype=bool)
    hop = round(m.BURST_HOP_SECONDS * 1000)
    diff_peak = m.frame_peaks(out - ref, hop)
    out_peak = m.frame_peaks(out, hop)

    result = m.null_test(diff_peak, out_peak, exclude_mask)

    assert not result.passed
    assert result.worst_residual_db > m.NULL_TEST_FAIL_DB
    assert result.pct_above_warn_db > 0


def test_null_test_ignores_a_perturbation_inside_the_excluded_mask():
    rng = np.random.default_rng(0)
    ref = rng.uniform(-0.5, 0.5, (1000, 2)).astype(np.float32)
    out = ref.copy()
    out[500:520] += 0.3
    exclude_mask = np.zeros(200, dtype=bool)
    exclude_mask[95:110] = True
    hop = round(m.BURST_HOP_SECONDS * 1000)
    diff_peak = m.frame_peaks(out - ref, hop)
    out_peak = m.frame_peaks(out, hop)

    result = m.null_test(diff_peak, out_peak, exclude_mask)

    assert result.passed


# --- trim / fade ---


def test_trim_fade_trims_to_the_requested_window():
    audio = np.ones((1000, 2), dtype=np.float32)
    out = m.trim_fade(audio, sr=100, start=1.0, end=9.0, fade_in=0.0, fade_out=0.0)
    assert out.shape == (800, 2)


def test_trim_fade_applies_linear_fade_edges():
    audio = np.ones((1000, 2), dtype=np.float32)
    out = m.trim_fade(audio, sr=100, start=None, end=None, fade_in=1.0, fade_out=1.0)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[500, 0] == pytest.approx(1.0)


def test_trim_fade_defaults_to_full_length_with_no_fades():
    audio = np.ones((1000, 2), dtype=np.float32)
    out = m.trim_fade(audio, sr=100, start=None, end=None, fade_in=0.0, fade_out=0.0)
    assert out.shape == audio.shape
    assert np.array_equal(out, audio)


# --- time code parsing ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("5", 5.0),
        ("5.5", 5.5),
        ("1:30", 90.0),
        ("1:30.5", 90.5),
        ("1:02:03", 3723.0),
    ],
)
def test_parse_timecode(value, expected):
    assert m.parse_timecode(value) == pytest.approx(expected)


def test_parse_timecode_rejects_invalid_format():
    with pytest.raises(ValueError):
        m.parse_timecode("1:2:3:4")


# --- bit depth / dither / final output stage ---


def test_validate_bit_depth_rejects_unsupported_values():
    with pytest.raises(ValueError):
        m.validate_bit_depth(32)


def test_finalize_output_dithers_only_at_16_bit(tmp_path):
    sr = 1000
    rendered16 = tmp_path / "rendered16.wav"
    rendered24 = tmp_path / "rendered24.wav"
    sf.write(rendered16, np.zeros((2000, 2), dtype=np.float64), sr, subtype="PCM_24")
    sf.write(rendered24, np.zeros((2000, 2), dtype=np.float64), sr, subtype="PCM_24")

    out16 = tmp_path / "out16.wav"
    out24 = tmp_path / "out24.wav"

    m.finalize_output(rendered16, out16, sr, None, None, 0.0, 0.0, bit_depth=16, move_when_possible=False)
    m.finalize_output(rendered24, out24, sr, None, None, 0.0, 0.0, bit_depth=24, move_when_possible=False)

    assert "16" in sf.info(out16).subtype
    assert "24" in sf.info(out24).subtype
    written16, _ = sf.read(out16)
    assert not np.all(written16 == 0)
    written24, _ = sf.read(out24)
    assert np.all(written24 == 0)


def test_finalize_output_moves_the_file_when_no_trim_fade_and_24_bit(tmp_path):
    sr = 1000
    rendered = tmp_path / "rendered.wav"
    sf.write(rendered, np.ones((100, 2), dtype=np.float64), sr, subtype="PCM_24")
    output = tmp_path / "out.wav"

    m.finalize_output(rendered, output, sr, None, None, 0.0, 0.0, bit_depth=24)

    assert not rendered.exists()
    assert output.exists()


def test_finalize_output_does_not_move_when_move_when_possible_is_false(tmp_path):
    sr = 1000
    rendered = tmp_path / "rendered.wav"
    sf.write(rendered, np.ones((100, 2), dtype=np.float64), sr, subtype="PCM_24")
    output = tmp_path / "out.wav"

    m.finalize_output(rendered, output, sr, None, None, 0.0, 0.0, bit_depth=24, move_when_possible=False)

    assert rendered.exists()
    assert output.exists()


def test_finalize_output_streams_trim_and_fade_matching_trim_fade(tmp_path):
    sr = 1000
    n = 2000
    rng = np.random.default_rng(1)
    audio = rng.uniform(-0.5, 0.5, (n, 2))
    rendered = tmp_path / "rendered.wav"
    sf.write(rendered, audio, sr, subtype="FLOAT")

    expected = m.trim_fade(audio, sr, start=0.5, end=1.5, fade_in=0.2, fade_out=0.3)

    output = tmp_path / "out.wav"
    m.finalize_output(rendered, output, sr, 0.5, 1.5, 0.2, 0.3, bit_depth=24, chunk_seconds=0.3)

    assert rendered.exists()
    result, _ = sf.read(output)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, atol=2e-6)


def test_run_ffmpeg_surfaces_ffmpeg_stderr_tail_on_failure(monkeypatch):
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    stderr = "ffmpeg version 9.0.1\n" + "config line\n" * 50 + "paglia.reference.wav: No space left on device\n"

    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(228, args, output="", stderr=stderr)

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as excinfo:
        m.measure_loudness(m.Path("in.wav"))
    message = str(excinfo.value)
    assert "exit 228" in message
    assert "No space left on device" in message
    assert "ffmpeg version" not in message


# --- K-weighting (ITU-R BS.1770-4 pre-filter) ---


def test_k_weighting_sos_matches_the_bs1770_48k_coefficient_tables():
    sos = m.k_weighting_sos(48000)

    shelf_b = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    shelf_a = [1.0, -1.69065929318241, 0.73248077421585]
    highpass_b = [1.0, -2.0, 1.0]
    highpass_a = [1.0, -1.99004745483398, 0.99007225036621]

    assert sos.shape == (2, 6)
    np.testing.assert_allclose(sos[0], shelf_b + shelf_a, atol=1e-6)
    np.testing.assert_allclose(sos[1], highpass_b + highpass_a, atol=1e-6)


def _stereo_wav(tmp_path, left, right, sr, name="in.wav", subtype="FLOAT"):
    path = tmp_path / name
    sf.write(path, np.stack([left, right], axis=1), sr, subtype=subtype)
    return path


def test_balance_meter_ignores_a_dc_offset_that_skews_raw_rms(tmp_path):
    sr, seconds = 16000, 10
    t = np.arange(seconds * sr) / sr
    tone = 0.1 * np.sin(2 * np.pi * 200 * t)
    path = _stereo_wav(tmp_path, tone + 0.05, tone, sr)

    diag = m.compute_diagnostics(path)

    raw_l = DB(np.sqrt(np.mean((tone + 0.05) ** 2)))
    raw_r = DB(np.sqrt(np.mean(tone**2)))
    assert raw_l - raw_r > 1.0
    assert np.abs(np.median(diag.k_rms_l_db - diag.k_rms_r_db)) < 0.1


def test_balance_meter_reports_a_broadband_level_imbalance(tmp_path):
    sr, seconds = 16000, 10
    t = np.arange(seconds * sr) / sr
    tone = 0.1 * np.sin(2 * np.pi * 200 * t)
    path = _stereo_wav(tmp_path, tone * 10 ** (2.3 / 20), tone, sr)

    diag = m.compute_diagnostics(path)

    assert np.median(diag.k_rms_l_db - diag.k_rms_r_db) == pytest.approx(2.3, abs=0.05)


def test_balance_curve_holds_the_correction_through_a_dip_below_the_dead_zone():
    sr, seconds = 1000, 600
    left_gain = np.full(seconds, 1.2)
    left_gain[200:400] = 0.6
    audio = sine_stereo(sr, seconds, left_gain, np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)

    assert curve.corr_db[250:350].mean() == pytest.approx(0.6, abs=0.1)
    assert curve.corr_db[100:500].min() > 0.4


def test_balance_curve_never_engages_when_the_imbalance_stays_below_the_dead_zone():
    sr, seconds = 1000, 600
    audio = sine_stereo(sr, seconds, np.full(seconds, 0.6), np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)

    assert np.abs(curve.corr_db).max() < 1e-9


# --- diagnostics: clipping threshold, channel count, polarity, DC ---


@pytest.mark.parametrize(
    "subtype,expected",
    [
        ("PCM_16", 32767 / 32768),
        ("PCM_24", (2**23 - 1) / 2**23),
        ("PCM_32", (2**31 - 1) / 2**31),
        ("FLOAT", 1.0),
        ("DOUBLE", 1.0),
    ],
)
def test_full_scale_threshold_follows_the_file_bit_depth(subtype, expected):
    assert m.full_scale_threshold(subtype) == pytest.approx(expected, rel=0, abs=1e-12)


def test_compute_diagnostics_counts_positive_full_scale_samples_in_24_bit_files(tmp_path):
    sr = 8000
    audio = np.zeros((sr, 2))
    audio[:100] = 1.0
    audio[100:200] = -1.0
    path = tmp_path / "clipped24.wav"
    sf.write(path, audio, sr, subtype="PCM_24")

    diag = m.compute_diagnostics(path)

    assert diag.clipped_samples == 400


def test_compute_diagnostics_rejects_a_mono_file(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.zeros(8000), 8000, subtype="PCM_16")

    with pytest.raises(ValueError, match="stereo"):
        m.compute_diagnostics(path)


def test_compute_diagnostics_flags_a_polarity_inverted_channel(tmp_path):
    sr, seconds = 16000, 10
    rng = np.random.default_rng(5)
    t = np.arange(seconds * sr) / sr
    music = 0.1 * np.sin(2 * np.pi * 200 * t) + 0.02 * rng.standard_normal(seconds * sr)
    path = _stereo_wav(tmp_path, music, -music, sr)

    diag = m.compute_diagnostics(path)
    warnings = m.diagnostics_warnings(diag)

    assert np.median(diag.lr_correlation) < -0.9
    assert any("polarity" in w for w in warnings)


def test_compute_diagnostics_is_quiet_on_a_healthy_stereo_file(tmp_path):
    sr, seconds = 16000, 10
    rng = np.random.default_rng(6)
    t = np.arange(seconds * sr) / sr
    music = 0.1 * np.sin(2 * np.pi * 200 * t) + 0.02 * rng.standard_normal(seconds * sr)
    path = _stereo_wav(tmp_path, music, music * 0.9, sr)

    diag = m.compute_diagnostics(path)

    assert np.median(diag.lr_correlation) > 0.9
    assert m.diagnostics_warnings(diag) == []


def test_compute_diagnostics_flags_a_dc_offset_above_the_report_threshold(tmp_path):
    sr, seconds = 16000, 5
    t = np.arange(seconds * sr) / sr
    tone = 0.1 * np.sin(2 * np.pi * 200 * t)
    path = _stereo_wav(tmp_path, tone + 0.01, tone, sr)

    diag = m.compute_diagnostics(path)
    warnings = m.diagnostics_warnings(diag)

    assert diag.dc_offset[0] == pytest.approx(0.01, abs=1e-4)
    assert any("DC" in w for w in warnings)


# --- render chain: integer 4x oversampling, double precision ---


@pytest.mark.parametrize("sr,oversampled", [(44100, 176400), (48000, 192000), (96000, 384000)])
def test_render_limited_oversamples_at_an_integer_multiple_of_the_source_rate(monkeypatch, tmp_path, sr, oversampled):
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    m.render_limited(tmp_path / "in.wav", tmp_path / "out.wav", gain_db=2.0, ceiling_db=-1.0, sample_rate=sr)

    stages = captured["args"][captured["args"].index("-af") + 1].split(",")
    assert stages[0] == "volume=2.0dB"
    assert stages[1].startswith(f"aresample={oversampled}")
    assert stages[2].startswith("alimiter=")
    assert stages[3].startswith(f"aresample={sr}")


# --- end to end with the real ffmpeg ---


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_full_chain_delivers_target_ceiling_null_and_balance(tmp_path):
    sr, seconds = 8000, 300
    target, ceiling = -14.0, -1.0
    rng = np.random.default_rng(7)
    t = np.arange(seconds * sr) / sr
    music = 0.15 * np.sin(2 * np.pi * 110 * t) + 0.05 * rng.standard_normal(seconds * sr)
    left_gain = np.ones(seconds)
    left_gain[60:240] = 10 ** (2.3 / 20)
    left = music * np.repeat(left_gain, sr)
    right = music.copy()
    for click_at in (20.0, 150.0, 280.0):
        s = int(click_at * sr)
        left[s : s + int(0.003 * sr)] = 0.95
    src = _stereo_wav(tmp_path, left, right, sr, name="src.wav", subtype="PCM_16")

    diag = m.compute_diagnostics(src)
    curve = m.balance_curve(diag.k_rms_l_db, diag.k_rms_r_db)
    assert curve.block_shape_ok
    assert curve.corr_db[100:200].mean() == pytest.approx(2.3, abs=0.2)

    stage = tmp_path / "stage.wav"
    m.apply_balance(src, stage, curve.corr_db)
    gain_db = m.compute_gain_db(target, m.measure_loudness(stage).integrated_lufs)

    rendered = tmp_path / "rendered.wav"
    m.render_limited(stage, rendered, gain_db, ceiling, sr)
    peak_ref, peak_out, diff_peak = m.stream_reference_diff(stage, rendered, gain_db, sr)
    touched = m.limiter_touched_mask(peak_ref, peak_out)
    radius = round(m.BURST_INFLUENCE_MS / (m.BURST_HOP_SECONDS * 1000))
    null = m.null_test(diff_peak, peak_out, m.dilate_mask(touched, radius))
    assert null.passed
    assert touched.any()

    output = tmp_path / "out.wav"
    m.finalize_output(rendered, output, sr, None, None, 0.0, 2.0, bit_depth=16)
    delivered = m.measure_loudness(output)
    assert m.verify_loudness_target(delivered.integrated_lufs, delivered.true_peak_dbfs, target, ceiling) == []

    after = m.compute_diagnostics(output)
    assert after.clipped_samples == 0
    diff = after.k_rms_l_db - after.k_rms_r_db
    assert np.abs(diff[100:200]).max() < 0.3
    assert np.abs(diff[10:50]).max() < 0.3


def test_balance_curve_does_not_let_the_first_second_dominate_the_opening_window():
    sr, seconds = 1000, 600
    left_gain = np.full(seconds, -2.5)
    left_gain[0] = 3.0
    audio = sine_stereo(sr, seconds, left_gain, np.zeros(seconds))
    rms_l_db, rms_r_db = per_second_rms_db(audio, sr)

    curve = m.balance_curve(rms_l_db, rms_r_db)

    assert curve.corr_db[:10].max() < -1.5
