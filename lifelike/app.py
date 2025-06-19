import click
import os
import subprocess
from diffusers import StableDiffusionPipeline
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset", "faces")
FINE_TUNED_DIR = os.path.join(DATASET_DIR, "..", "output", "finetuned")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
DREAMBOOTH_SCRIPT = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "external",
        "diffusers",
        "examples",
        "dreambooth",
        "train_dreambooth_lora_sdxl.py",
    )
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe = get_pipe()
    prompt = (
        "portrait photo of a young person, highly detailed, photorealistic, sharp focus, "
        "centered, studio lighting, looking at camera, neutral background"
    )
    generator = torch.Generator(device=pipe.device).manual_seed(
        hash(person_id) % (2**32)
    )
    image = pipe(
        prompt, num_inference_steps=40, guidance_scale=10.0, generator=generator
    ).images[0]
    out_path = os.path.join(OUTPUT_DIR, f"{person_id}_base.png")
    image.save(out_path)
    click.echo(f"Generated base face for '{person_id}' at {out_path}")


@cli.command()
@click.argument("person_id")
@click.argument("instance_data_dir")
@click.option("--output_dir", default=None, help="Where to save the fine-tuned model")
@click.option(
    "--steps",
    default=1600,
    help="Number of training steps (recommend 1200-2000 for best results)",
)
@click.option(
    "--prior_loss_weight",
    default=1.0,
    help="Prior loss weight for regularization (try 1.0-1.5 if overfitting)",
)
def fine_tune_person(
    person_id, instance_data_dir, output_dir, steps, prior_loss_weight
):
    """
    Fine-tune a model for PERSON_ID using DreamBooth.
    INSTANCE_DATA_DIR should contain images of the person.
    """
    if output_dir is None:
        output_dir = os.path.join(FINE_TUNED_DIR, f"{person_id}_finetuned")
    os.makedirs(output_dir, exist_ok=True)
    pretrained_model = "stabilityai/stable-diffusion-xl-base-1.0"
    instance_prompt = f"a photo of {person_id} person"
    command = [
        "accelerate",
        "launch",
        "--mixed_precision",
        "no",
        DREAMBOOTH_SCRIPT,
        "--pretrained_model_name_or_path",
        pretrained_model,
        "--instance_data_dir",
        instance_data_dir,
        "--output_dir",
        output_dir,
        "--instance_prompt",
        instance_prompt,
        "--resolution",
        "512",
        "--train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "5e-6",
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--max_train_steps",
        str(steps),
        "--checkpointing_steps",
        "100",
        "--prior_loss_weight",
        str(prior_loss_weight),
    ]
    click.echo(f"Running DreamBooth fine-tuning for '{person_id}'...")
    subprocess.run(command, check=True)
    click.echo(f"Fine-tuned model for '{person_id}' saved at {output_dir}")


@cli.command()
@click.argument("person_id")
@click.option("--count", default=5, help="Number of variations to generate")
def generate_variations(person_id, count):
    """Generate multiple high-quality studio-like images for PERSON_ID using the fine-tuned model (if available)."""
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


@cli.command()
def version():
    """Show the version of LifeLike CLI."""
    click.echo("LifeLike CLI version 0.1.0")


if __name__ == "__main__":
    cli()
