import click
import os
import torch
from lifelike.generate_person import get_pipe


@click.command()
@click.argument("person_id")
@click.option("--count", default=5, help="Number of variations to generate")
def generate_variations(person_id, count):
    """Generate multiple high-quality studio-like images for PERSON_ID using the fine-tuned model (if available)."""
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
    FINE_TUNED_DIR = os.path.join(OUTPUT_DIR, "finetuned")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fine_tuned_path = os.path.join(FINE_TUNED_DIR, f"{person_id}_finetuned")
    if os.path.exists(fine_tuned_path):
        pipe = get_pipe(fine_tuned_path)
    else:
        pipe = get_pipe()
    base_seed = hash(person_id) % (2**32)
    prompts = [
        f"ultra high resolution, studio portrait photo of {person_id} person, smiling, highly detailed, photorealistic, sharp focus, centered, studio lighting, looking at camera, clean background",
        f"ultra high resolution, studio portrait photo of {person_id} person, side view, highly detailed, photorealistic, sharp focus, centered, studio lighting, looking at camera, clean background",
        f"ultra high resolution, studio portrait photo of {person_id} person, wearing glasses, highly detailed, photorealistic, sharp focus, centered, studio lighting, looking at camera, clean background",
        f"ultra high resolution, studio portrait photo of {person_id} person, laughing, highly detailed, photorealistic, sharp focus, centered, studio lighting, looking at camera, clean background",
        f"ultra high resolution, studio portrait photo of {person_id} person, outdoors, highly detailed, photorealistic, sharp focus, centered, studio lighting, looking at camera, clean background",
    ]
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        generator = torch.Generator(device=pipe.device).manual_seed(base_seed + i)
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=12.0,
            generator=generator,
            height=768,
            width=768,
        ).images[0]
        out_path = os.path.join(OUTPUT_DIR, f"{person_id}_var_{i+1:02d}.png")
        image.save(out_path)
        click.echo(
            f"Saved high-quality variation {i+1} for '{person_id}' at {out_path}"
        )
