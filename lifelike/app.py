import click
import os
from diffusers import StableDiffusionPipeline
import torch

DATASET_DIR = os.path.join(os.path.dirname(__file__), "lifelike", "dataset", "ai_faces")
FINE_TUNED_DIR = os.path.join(
    os.path.dirname(__file__), "lifelike", "dataset", "ai_faces", "finetuned"
)


def get_pipe(model_path=None):
    if model_path is None:
        model_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe


@click.group()
def cli():
    """LifeLike AI Avatar CLI - Generate AI-powered videos with synthetic avatars and backgrounds."""
    pass


@cli.command()
@click.argument("person_id")
def generate_person(person_id):
    """Generate a new AI face for PERSON_ID and save it."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    pipe = get_pipe()
    prompt = f"portrait photo of a young person, realistic, looking at camera, neutral background"
    generator = torch.Generator(device=pipe.device).manual_seed(
        hash(person_id) % (2**32)
    )
    image = pipe(
        prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator
    ).images[0]
    out_path = os.path.join(DATASET_DIR, f"{person_id}_base.png")
    image.save(out_path)
    click.echo(f"Generated base face for '{person_id}' at {out_path}")


@cli.command()
@click.argument("person_id")
def fine_tune_person(person_id):
    """Placeholder: Fine-tune a model for PERSON_ID using DreamBooth/LoRA."""
    # In a real implementation, you would call a DreamBooth/LoRA training script here.
    # For now, just simulate the process and create a directory for the fine-tuned model.
    os.makedirs(FINE_TUNED_DIR, exist_ok=True)
    # Simulate fine-tuning by copying the base model path (in real use, train and save new weights)
    fine_tuned_path = os.path.join(FINE_TUNED_DIR, f"{person_id}_finetuned")
    with open(fine_tuned_path, "w") as f:
        f.write(
            "This is a placeholder for the fine-tuned model for person_id: " + person_id
        )
    click.echo(
        f"[Placeholder] Fine-tuned model for '{person_id}' would be saved at {fine_tuned_path}"
    )


@cli.command()
@click.argument("person_id")
@click.option("--count", default=5, help="Number of variations to generate")
def generate_variations(person_id, count):
    """Generate multiple images for PERSON_ID using the fine-tuned model (if available)."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    fine_tuned_path = os.path.join(FINE_TUNED_DIR, f"{person_id}_finetuned")
    # In a real implementation, check if fine-tuned model exists and use it
    # For now, always use the base model
    pipe = get_pipe()  # Would be get_pipe(fine_tuned_path) if real fine-tune
    base_seed = hash(person_id) % (2**32)
    prompts = [
        "portrait photo of a young person, smiling, realistic, neutral background",
        "portrait photo of a young person, side view, realistic, neutral background",
        "portrait photo of a young person, wearing glasses, realistic, neutral background",
        "portrait photo of a young person, laughing, realistic, neutral background",
        "portrait photo of a young person, outdoors, realistic, neutral background",
    ]
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        generator = torch.Generator(device=pipe.device).manual_seed(base_seed + i)
        image = pipe(
            prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator
        ).images[0]
        out_path = os.path.join(DATASET_DIR, f"{person_id}_var_{i+1:02d}.png")
        image.save(out_path)
        click.echo(f"Saved variation {i+1} for '{person_id}' at {out_path}")


@cli.command()
def version():
    """Show the version of LifeLike CLI."""
    click.echo("LifeLike CLI version 0.1.0")


if __name__ == "__main__":
    cli()
