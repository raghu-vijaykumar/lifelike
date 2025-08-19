import click
import torch
import json
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


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


@click.command()
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
