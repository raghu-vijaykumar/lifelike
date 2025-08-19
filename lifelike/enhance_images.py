import click
import glob
import os
import numpy as np
from PIL import Image


@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--method",
    type=click.Choice(["gfpgan", "codeformer"]),
    default="gfpgan",
    help="Enhancement method to use.",
)
def enhance_images(folder, method):
    """
    Enhance all PNG images in the given FOLDER using GFPGAN or CodeFormer.
    Saves enhanced images with _enhancer.png suffix.
    """
    image_paths = glob.glob(os.path.join(folder, "*.png"))
    if method == "gfpgan":
        from gfpgan import GFPGANer

        restorer = GFPGANer(
            model_path=None,  # Use default weights
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            _, _, enhanced_img = restorer.enhance(
                np.array(img), has_aligned=False, only_center_face=True, paste_back=True
            )
            out_path = img_path.replace(".png", "_gfpgan_enhanced.png")
            Image.fromarray(enhanced_img).save(out_path)
            click.echo(
                f"Enhanced (GFPGAN): {os.path.basename(img_path)} -> {os.path.basename(out_path)}"
            )
    elif method == "codeformer":
        try:
            import torch
            from basicsr.archs.codeformer_arch import CodeFormer
            from torchvision.transforms.functional import to_tensor, to_pil_image
        except ImportError:
            click.echo(
                "CodeFormer or dependencies not installed. Please install basicsr and CodeFormer."
            )
            return
        model_path = os.path.expanduser("weights/CodeFormer/codeformer.pth")
        if not os.path.exists(model_path):
            click.echo(
                f"CodeFormer weights not found at {model_path}. Please download them."
            )
            return
        net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).eval()
        net.load_state_dict(torch.load(model_path, map_location="cpu")["params_ema"])
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = to_tensor(img).unsqueeze(0)
            with torch.no_grad():
                enhanced_tensor = net(img_tensor)[0]
            enhanced_img = to_pil_image(enhanced_tensor.squeeze(0).clamp(0, 1))
            out_path = img_path.replace(".png", "_codeformer_enhanced.png")
            enhanced_img.save(out_path)
            click.echo(
                f"Enhanced (CodeFormer): {os.path.basename(img_path)} -> {os.path.basename(out_path)}"
            )
    click.echo("Enhancement complete.")
