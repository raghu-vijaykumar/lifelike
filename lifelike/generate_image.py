import click
import os
import torch
from lifelike.generate_person import get_pipe


@click.command(name="generate-image")
@click.argument("prompt", required=False)
@click.option(
    "--prompt-file",
    type=click.Path(exists=True),
    help="Path to a file with one prompt per line for batch generation.",
)
def generate_image(prompt, prompt_file):
    """
    Generate an AI image from a prompt or a batch of prompts (one per line in a file).
    If using a prompt file, each output will be named by its line number.
    """
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe = get_pipe()
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        for idx, line_prompt in enumerate(prompts, 1):
            generator = torch.Generator(device=pipe.device).manual_seed(
                hash(line_prompt) % (2**32)
            )
            image = pipe(
                line_prompt,
                num_inference_steps=40,
                guidance_scale=10.0,
                generator=generator,
            ).images[0]
            out_path = os.path.join(OUTPUT_DIR, f"image_{idx:02d}.png")
            image.save(out_path)
            click.echo(f"Generated image for line {idx}: '{line_prompt}' at {out_path}")
    elif prompt:
        generator = torch.Generator(device=pipe.device).manual_seed(
            hash(prompt) % (2**32)
        )
        image = pipe(
            prompt,
            num_inference_steps=40,
            guidance_scale=10.0,
            generator=generator,
        ).images[0]
        out_path = os.path.join(OUTPUT_DIR, "image_01.png")
        image.save(out_path)
        click.echo(f"Generated image for prompt '{prompt}' at {out_path}")
    else:
        click.echo("Please provide a prompt or a --prompt-file")
