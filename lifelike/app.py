import itertools
import json
import random
import click
import os
import subprocess
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import secrets

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
        model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    if "xl" in model_path or "sdxl" in model_path:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        )
    else:
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


import json
import itertools
import random


@cli.command()
@click.argument("person_id")
@click.option("--per-angle", default=5, help="Images per unique combination")
@click.option(
    "--gender", default="female", type=click.Choice(["male", "female", "neutral"])
)
@click.option("--ethnicity", default="indian")
@click.option(
    "--age", default="adult", type=click.Choice(["child", "teen", "adult", "elderly"])
)
@click.option(
    "--moods", default="neutral,smiling,serious,laughing,angry,curious,confused"
)
@click.option("--angles", default="front,side,profile,3/4 view,head tilt")
@click.option("--styles", default="traditional saree,modern casual,formal suit,kurtis")
@click.option(
    "--poses", default="standing,sitting,arms crossed,hand on chin,looking up"
)
@click.option("--accessories", default="none,earrings,glasses,watch,nose ring,bindi")
@click.option(
    "--lighting",
    default="soft studio,dramatic rim,warm sunset,cool white,diffused softbox",
)
@click.option(
    "--backgrounds",
    default="plain white,gradient gray,indoor room,park,urban street,nature forest",
)
@click.option("--resolution", default="1080*1344")
@click.option("--seed-base", default=12345)
@click.option("--expression-intensity", default="mild")
@click.option("--output-dir", default="output/generated_person")
@click.option("--zip-output", is_flag=True, default=True)
@click.option("--face-focus", is_flag=True, default=True)
@click.option("--include-full-body", is_flag=True, default=False)
@click.option("--camera-type", default="DSLR")
@click.option(
    "--hair-styles",
    default="long straight,curly bun,braided,short wavy,shoulder length",
)
@click.option(
    "--max-photos",
    default=None,
    type=int,
    help="Maximum number of images to generate (optional)",
)
def generate_person(
    person_id,
    per_angle,
    gender,
    ethnicity,
    age,
    moods,
    angles,
    styles,
    poses,
    accessories,
    lighting,
    backgrounds,
    resolution,
    seed_base,
    expression_intensity,
    output_dir,
    zip_output,
    face_focus,
    include_full_body,
    camera_type,
    hair_styles,
    max_photos,
):
    from PIL import Image
    from pathlib import Path

    if include_full_body and face_focus:
        click.echo(
            "⚠️ Both --include-full-body and --face-focus are set. Prioritizing full-body and disabling face focus."
        )
        face_focus = False
    elif not include_full_body and not face_focus:
        click.echo(
            "ℹ️ Neither --include-full-body nor --face-focus set. Defaulting to face-focused portraits."
        )
        face_focus = True

    width, height = map(int, resolution.lower().split("x"))
    person_dir = Path(output_dir) / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    pipe = get_pipe("stabilityai/stable-diffusion-xl-base-1.0")

    mood_list = [x.strip() for x in moods.split(",") if x.strip()]
    angle_list = [x.strip() for x in angles.split(",") if x.strip()]
    style_list = [x.strip() for x in styles.split(",") if x.strip()]
    pose_list = [x.strip() for x in poses.split(",") if x.strip()]
    accessory_list = [x.strip() for x in accessories.split(",") if x.strip()]
    lighting_list = [x.strip() for x in lighting.split(",") if x.strip()]
    background_list = [x.strip() for x in backgrounds.split(",") if x.strip()]
    hair_style_list = [x.strip() for x in hair_styles.split(",") if x.strip()]

    combinations = []

    combinations = list(
        itertools.product(
            angle_list,
            mood_list,
            style_list,
            pose_list,
            accessory_list,
            lighting_list,
            background_list,
            hair_style_list,
        )
    )

    if max_photos is not None:
        max_combos = max_photos // per_angle
        if max_combos < len(combinations):
            click.echo(
                f"🎯 Limiting to {max_photos} images by randomly selecting {max_combos} combinations."
            )
            random.seed(seed_base)
            combinations = random.sample(combinations, max_combos)
    else:
        click.echo(
            f"🔄 Generating all {len(combinations)} unique combinations with {per_angle} images each."
        )

    metadata = []

    for idx, (
        angle,
        mood,
        style,
        pose,
        accessory,
        light,
        background,
        hair,
    ) in enumerate(combinations):
        for i in range(per_angle):
            full_prompt = (
                f"A {gender} {ethnicity} person with {hair}, {age}, {mood} expression, "
                f"{expression_intensity} intensity, in {style}, {pose}, {accessory}, {angle}, {light} lighting, "
                f"background: {background}, shot with {camera_type}, photorealistic, {resolution}, "
                f"{'full body' if include_full_body else 'portrait'}, {'focused on face' if face_focus else ''}"
            )

            seed = seed_base + idx * 100 + i
            generator = torch.Generator(device=pipe.device).manual_seed(seed)

            image = pipe(
                prompt=full_prompt,
                num_inference_steps=45,
                guidance_scale=10.0,
                generator=generator,
                height=height,
                width=width,
            ).images[0]

            filename = f"{person_id}_{idx:04d}_{i+1:02d}.png"
            filepath = person_dir / filename
            image.save(filepath)

            image_metadata = {
                "filename": filename,
                "prompt": full_prompt,
                "seed": seed,
                "angle": angle,
                "mood": mood,
                "style": style,
                "pose": pose,
                "accessory": accessory,
                "lighting": light,
                "background": background,
                "hair": hair,
            }
            meta_filepath = person_dir / f"{person_id}_{idx:04d}_{i+1:02d}.json"
            with open(meta_filepath, "w") as meta_f:
                json.dump(image_metadata, meta_f, indent=2)

            metadata.append(image_metadata)

    meta_path = person_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_images = len(combinations) * per_angle
    click.echo(f"✅ Total images generated for '{person_id}': {total_images}")
    click.echo(f"📁 Saved to: {person_dir}")
    click.echo(f"📝 Metadata file: {meta_path}")

    if zip_output:
        import shutil

        zip_path = str(person_dir) + ".zip"
        shutil.make_archive(str(person_dir), "zip", str(person_dir))
        click.echo(f"📦 Zipped output: {zip_path}")


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


@cli.command(name="generate-image")
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


@cli.command()
def version():
    """Show the version of LifeLike CLI."""
    click.echo("LifeLike CLI version 0.1.0")


@cli.command()
@click.argument("person_id")
@click.option(
    "--gender", default="female", type=click.Choice(["male", "female", "neutral"])
)
@click.option("--ethnicity", default="indian")
@click.option(
    "--age", default="adult", type=click.Choice(["child", "teen", "adult", "elderly"])
)
@click.option(
    "--moods", default="neutral,smiling,serious,laughing,angry,curious,confused"
)
@click.option("--angles", default="front,side,profile,3/4 view,head tilt")
@click.option(
    "--lighting",
    default="soft studio,dramatic rim,warm sunset,cool white,diffused softbox",
)
@click.option("--resolution", default="1080x1344")
@click.option("--seed-base", default=12345)
@click.option("--output-dir", default="output/generated_person_dreambooth")
@click.option("--face-focus", is_flag=True, default=True)
def generate_dreamboothset(
    person_id,
    gender,
    ethnicity,
    age,
    moods,
    angles,
    lighting,
    resolution,
    seed_base,
    output_dir,
    face_focus,
):
    """
    Generate a DreamBooth dataset for PERSON_ID, controlling only angle, lighting, and moods.
    Generates all combinations of angles, moods, and lighting.
    """
    from PIL import Image
    from pathlib import Path
    import torch
    import json

    width, height = map(int, resolution.lower().split("x"))
    person_dir = Path(output_dir) / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    pipe = get_pipe("stabilityai/stable-diffusion-xl-base-1.0")

    mood_list = [x.strip() for x in moods.split(",") if x.strip()]
    angle_list = [x.strip() for x in angles.split(",") if x.strip()]
    lighting_list = [x.strip() for x in lighting.split(",") if x.strip()]

    total_to_generate = len(angle_list) * len(mood_list) * len(lighting_list)
    click.echo(
        f"Preparing to generate {total_to_generate} images for DreamBooth set..."
    )
    metadata = []
    idx = 0
    for angle in angle_list:
        for mood in mood_list:
            for light in lighting_list:
                full_prompt = (
                    f"A {gender} {ethnicity}, "
                    f"{'hot and handsome' if gender.lower() == 'male' else 'hot and beautiful'}, "
                    f"{age}, {mood} expression, {angle}, {light} lighting, "
                    f"{'chiseled features, sharp jawline, stylish hair' if gender.lower() == 'male' else 'flawless skin, captivating eyes, silky hair'}, "
                    f"portrait, {'focused on face' if face_focus else ''}, ultra high resolution."
                )

                seed = seed_base + idx
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                image = pipe(
                    prompt=full_prompt,
                    num_inference_steps=45,
                    guidance_scale=10.0,
                    generator=generator,
                    height=height,
                    width=width,
                ).images[0]
                filename = f"{person_id}_dreambooth_{idx:04d}.png"
                filepath = person_dir / filename
                image.save(filepath)
                image_metadata = {
                    "filename": filename,
                    "prompt": full_prompt,
                    "seed": seed,
                    "angle": angle,
                    "mood": mood,
                    "lighting": light,
                }
                meta_filepath = person_dir / f"{person_id}_dreambooth_{idx:04d}.json"
                with open(meta_filepath, "w") as meta_f:
                    json.dump(image_metadata, meta_f, indent=2)
                metadata.append(image_metadata)
                idx += 1
    meta_path = person_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    click.echo(f"✅ DreamBooth set generated for '{person_id}': {idx} images")
    click.echo(f"📁 Saved to: {person_dir}")
    click.echo(f"📝 Metadata file: {meta_path}")
    click.echo(f"Total photos generated: {idx}")


if __name__ == "__main__":
    cli()
