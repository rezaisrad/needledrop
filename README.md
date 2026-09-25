# needledrop

Masters a vinyl DJ set recording. It corrects L/R balance per deck, sets loudness to a LUFS target with a true-peak limiter, and can trim and fade the ends.

It does not filter the audio: no high-pass, hum, hiss or declick filters, no click repair, no DC removal. Clicks are listed in the report and go through the limiter with everything else. DC offset and polarity are reported but not changed.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- ffmpeg on your PATH
- A stereo recording (WAV, FLAC, AIFF)

## Install

```sh
uvx --from git+https://github.com/rezaisrad/needledrop needledrop my-set.wav
```

or

```sh
uv tool install git+https://github.com/rezaisrad/needledrop
```

## Usage

```sh
needledrop my-set.wav                    # writes my-set_master.wav: -14 LUFS, -1 dBTP, 24-bit
needledrop my-set.wav --dry-run          # report only, writes nothing
needledrop my-set.wav --target -16 -o upload.wav
needledrop my-set.wav --start 0:12 --end 2:14:30 --fade-in 2 --fade-out 8
```

| Option | Default | |
|---|---|---|
| `-o, --output` | `<input>_master.wav` | Output file |
| `--target` | `-14` | Integrated loudness (LUFS) |
| `--ceiling` | `-1` | True-peak ceiling (dBTP) |
| `--balance / --no-balance` | on | L/R balance correction |
| `--dry-run` | off | Report only |
| `--keep-stages` | off | Keep intermediate files |
| `--bit-depth` | `24` | `24` or `16` (16 is dithered) |
| `--start`, `--end` | | Trim, as seconds or `mm:ss` / `h:mm:ss` |
| `--fade-in`, `--fade-out` | `0` | Linear fade, seconds |

The input file is never changed.

## How it works

1. Measures the source: peak, true peak, loudness, clipped samples, DC offset, L/R correlation, and clicks.
2. Balance: measures the L/R level difference each second (K-weighted, BS.1770), smooths it with a 61 s median, and corrects anything over 0.8 dB, up to ±3 dB. Correction stops once the difference drops below 0.4 dB.
3. Applies one gain to reach the target, then limits at 4x the sample rate to catch inter-sample peaks.
4. Renders the same chain without the limiter and compares. Outside the spots where the limiter acted, the two must match within -50 dB or the run fails. It warns if the limiter cut more than 6 dB or touched more than 2% of the set.
5. Trims, fades and writes the file, then measures it again. If loudness or true peak is off, the output is deleted and the command exits with an error.

Audio is processed in 1 s chunks, so memory use stays low for long sets. Intermediate files need about 3x the input size in free disk space.

## Development

```sh
uv sync
uv run pytest
```

One test runs real ffmpeg and is skipped if ffmpeg isn't installed.

`scripts/probe_resampler_roundtrip.py` measures the resampler settings used by the limiter:

```sh
uv run python scripts/probe_resampler_roundtrip.py --rates 44100,48000
```

## License

MIT
