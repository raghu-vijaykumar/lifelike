import click
import random
import itertools
import json
from pathlib import Path


def get_pipe(model_path=None):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

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
    import torch

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
