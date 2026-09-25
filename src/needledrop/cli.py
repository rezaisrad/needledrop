from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from needledrop import mastering

app = typer.Typer()
console = Console()


@app.command()
def master_set(
    input_path: Path = typer.Argument(..., help="Recorded vinyl DJ set to master"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output file (default: <input>_master.wav next to input)"
    ),
    target: float = typer.Option(
        mastering.DEFAULT_TARGET_LUFS, "--target", help="Target integrated loudness (LUFS)"
    ),
    ceiling: float = typer.Option(
        mastering.DEFAULT_CEILING_DBTP, "--ceiling", help="True-peak ceiling (dBTP)"
    ),
    balance: bool = typer.Option(
        True, "--balance/--no-balance", help="Apply per-deck L/R balance correction"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Diagnostics and burst report only; writes nothing"
    ),
    keep_stages: bool = typer.Option(
        False, "--keep-stages", help="Keep intermediate stage files instead of deleting them"
    ),
    bit_depth: int = typer.Option(24, "--bit-depth", help="Output bit depth: 24 or 16"),
    start: str = typer.Option(None, "--start", help="Trim start (seconds or mm:ss / h:mm:ss)"),
    end: str = typer.Option(None, "--end", help="Trim end (seconds or mm:ss / h:mm:ss)"),
    fade_in: float = typer.Option(0.0, "--fade-in", help="Fade-in duration in seconds"),
    fade_out: float = typer.Option(0.0, "--fade-out", help="Fade-out duration in seconds"),
):
    """Master a recorded vinyl DJ set.

    Trim/fade, per-deck L/R balance correction, loudness normalization to a
    LUFS target with a true-peak limiter. No blanket vinyl clean-up filters
    and no click filling: record clicks are only detected and reported, then
    capped by the limiter like the rest of the audio.
    """
    if not input_path.exists():
        console.print(f"[red]Path not found: {input_path}[/red]")
        raise typer.Exit(1)

    try:
        mastering.validate_bit_depth(bit_depth)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    try:
        mastering.ensure_ffmpeg_available()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    try:
        start_s = mastering.parse_timecode(start) if start is not None else None
        end_s = mastering.parse_timecode(end) if end is not None else None
    except ValueError as e:
        console.print(f"[red]Invalid time code: {e}[/red]")
        raise typer.Exit(1)

    output_path = output if output is not None else input_path.with_name(f"{input_path.stem}_master.wav")
    stage_path = output_path.with_name(f"{output_path.stem}.stage.wav")
    rendered_path = output_path.with_name(f"{output_path.stem}.rendered.wav")

    console.print(f"[cyan]Diagnostics: {input_path.name}[/cyan]")
    before_diag = mastering.compute_diagnostics(input_path)
    sr = before_diag.samplerate
    before_loud = mastering.measure_loudness(input_path)
    before_snapshot = mastering.Snapshot(
        before_diag, before_loud.integrated_lufs, before_loud.true_peak_dbfs, before_loud.lra
    )
    console.print(mastering.build_diagnostics_table("Source diagnostics", before_snapshot))
    for warning in mastering.diagnostics_warnings(before_diag):
        console.print(f"[yellow]{warning}[/yellow]")

    bursts = mastering.detect_bursts(
        before_diag.frame_peak,
        before_diag.frame_peak_l,
        before_diag.frame_peak_r,
        before_diag.frame_mean_square,
        sr,
        before_diag.per_second_peak,
    )
    burst_table, extra = mastering.build_burst_table(bursts)
    console.print(burst_table)
    if extra:
        console.print(f"[dim]... and {extra} more burst(s) not shown[/dim]")

    if dry_run:
        return

    if balance:
        curve = mastering.balance_curve(before_diag.k_rms_l_db, before_diag.k_rms_r_db)
        console.print(mastering.build_balance_table(curve))
        if curve.block_shape_ok:
            corr_db = curve.corr_db
        else:
            console.print(
                "[yellow]Balance correction is not block-shaped; skipping the balance stage.[/yellow]"
            )
            corr_db = np.zeros(before_diag.num_seconds)
    else:
        console.print("[dim]Balance correction disabled (--no-balance)[/dim]")
        corr_db = np.zeros(before_diag.num_seconds)

    try:
        if np.any(corr_db):
            mastering.apply_balance(input_path, stage_path, corr_db)
            chain_input = stage_path
        else:
            chain_input = input_path

        gain_db = mastering.compute_gain_db(target, mastering.measure_loudness(chain_input).integrated_lufs)

        console.print("[cyan]Rendering...[/cyan]")
        mastering.render_limited(chain_input, rendered_path, gain_db, ceiling, sr)
        peak_ref, peak_out, diff_peak = mastering.stream_reference_diff(chain_input, rendered_path, gain_db, sr)

        if not keep_stages:
            stage_path.unlink(missing_ok=True)

        touched = mastering.limiter_touched_mask(peak_ref, peak_out)
        radius_frames = round(mastering.BURST_INFLUENCE_MS / (mastering.BURST_HOP_SECONDS * 1000))
        touched_dilated = mastering.dilate_mask(touched, radius_frames)

        null_result = mastering.null_test(diff_peak, peak_out, touched_dilated)
        if not null_result.passed:
            console.print(
                f"[red]Null test failed: worst residual {null_result.worst_residual_db:.1f} dB outside "
                "limiter influence — something other than the limiter changed the audio.[/red]"
            )
            raise typer.Exit(1)

        acc = mastering.limiter_accounting(
            peak_ref, peak_out, ceiling, bursts, sr, mastering.BURST_HOP_SECONDS, target
        )
        console.print(mastering.build_limiter_table(acc))
        console.print(mastering.format_null_test_line(null_result))

        mastering.finalize_output(
            rendered_path,
            output_path,
            sr,
            start_s,
            end_s,
            fade_in,
            fade_out,
            bit_depth,
            move_when_possible=not keep_stages,
        )

        delivered = mastering.measure_loudness(output_path)
        after_diag = mastering.compute_diagnostics(output_path)
        after_snapshot = mastering.Snapshot(
            after_diag, delivered.integrated_lufs, delivered.true_peak_dbfs, delivered.lra
        )
        console.print(mastering.build_before_after_table(before_snapshot, after_snapshot))

        violations = mastering.verify_loudness_target(
            delivered.integrated_lufs, delivered.true_peak_dbfs, target, ceiling
        )
        if violations:
            for violation in violations:
                console.print(f"[red]{violation}[/red]")
            output_path.unlink(missing_ok=True)
            console.print("[red]Output removed.[/red]")
            raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    finally:
        if not keep_stages:
            stage_path.unlink(missing_ok=True)
            rendered_path.unlink(missing_ok=True)

    console.print(f"[green]Output:[/green] {output_path}")


if __name__ == "__main__":
    app()
