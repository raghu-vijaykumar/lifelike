import click
from pathlib import Path
import os
import subprocess
import sys


@click.command()
@click.option(
    "--image-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the input image.",
)
@click.option(
    "--script", required=True, type=str, help="Script text or path to a text file."
)
@click.option(
    "--voice", default=None, type=str, help="Voice style or TTS model (optional)."
)
@click.option(
    "--background",
    default="studio",
    type=str,
    help="Background style (default: studio).",
)
@click.option(
    "--output",
    default="output/animated_video.mp4",
    type=click.Path(),
    help="Output video path.",
)
def animate_image(image_path, script, voice, background, output):
    """
    Animate the provided image with lip sync and studio effects for a given script using SadTalker.
    """
    # Resolve script text
    script_path = Path(script)
    if script_path.exists():
        with open(script_path, "r", encoding="utf-8") as f:
            script_text = f.read()
    else:
        script_text = script

    # Step 1: Convert script to audio (TTS)
    tts_output = Path(output).with_suffix(".wav")
    # Example using edge-tts (install with: pip install edge-tts)
    try:
        import edge_tts
        import asyncio

        async def tts_async(text, out_path, voice):
            communicate = edge_tts.Communicate(text, voice=voice or "en-US-AriaNeural")
            await communicate.save(out_path)

        asyncio.run(tts_async(script_text, str(tts_output), voice))
        click.echo(f"✅ TTS audio generated: {tts_output}")
    except ImportError:
        click.echo(
            "⚠️ edge-tts not installed. Please install with 'pip install edge-tts'."
        )
        return

    # Step 2: Run SadTalker animation
    # Assumes SadTalker repo is cloned to ./SadTalker and requirements are installed
    sadtalker_dir = Path("external/SadTalker")
    sadtalker_checkpoints = sadtalker_dir / "checkpoints"
    if not sadtalker_checkpoints.exists():
        click.echo(
            f"❌ SadTalker checkpoints not found at {sadtalker_checkpoints}. Please run 'python init.py' to set up."
        )
        return
    sadtalker_script = sadtalker_dir / "inference.py"
    if not sadtalker_script.exists():
        click.echo(
            f"❌ SadTalker not found at {sadtalker_script}. Please clone SadTalker repo."
        )
        return

    # Prepare command
    cmd = [
        sys.executable,
        str(sadtalker_script),
        "--driven_audio",
        str(tts_output),
        "--source_image",
        str(image_path),
        "--checkpoint_dir",
        str(sadtalker_checkpoints),
        "--result_dir",
        str(Path(output).parent),
        "--still",
        "--preprocess",
        "full",
        "--enhancer",
        "gfpgan",
    ]
    click.echo(f"Running SadTalker: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        click.echo(f"✅ Animation complete. Video saved in {Path(output).parent}")
    else:
        click.echo(f"❌ SadTalker failed: {result.stderr}")

    # Placeholder: Call SadTalker pipeline here
    # Example: sad_talker_animate(image_path, script_text, voice, background, output)
    click.echo(f"Animating image: {image_path}")
    click.echo(f"Script: {script_text[:60]}{'...' if len(script_text) > 60 else ''}")
    click.echo(f"Voice: {voice}")
    click.echo(f"Background: {background}")
    click.echo(f"Output video will be saved to: {output}")
    # TODO: Integrate SadTalker inference here
    click.echo("✅ Animation complete (placeholder)")
