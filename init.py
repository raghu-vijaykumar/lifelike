import os
import sys
import subprocess
import venv
import urllib.request
from pathlib import Path


def run(cmd, cwd=None):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)


def create_venv(venv_dir, requirements, torch_cuda=False):
    venv_path = Path(venv_dir)
    pip_path = venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip"
    activate_path = (
        venv_path
        / ("Scripts" if os.name == "nt" else "bin")
        / ("activate.bat" if os.name == "nt" else "activate")
    )
    if not venv_path.exists():
        print(f"Creating venv: {venv_dir}")
        venv.create(venv_dir, with_pip=True)
    else:
        print(f"Venv already exists: {venv_dir}")
    # Install torch/torchvision/torchaudio first
    if torch_cuda:
        # Stable Diffusion: install latest torch/torchvision/torchaudio with CUDA 11.8
        run(
            f'"{pip_path}" install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
        )
    else:
        # SadTalker: install torch==1.12.1+cu113, torchvision==0.13.1+cu113, torchaudio==0.12.1 with CUDA 11.3
        run(
            f'"{pip_path}" install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113'
        )
    # Install all other requirements from requirements file (default PyPI)
    run(f'"{pip_path}" install --upgrade -r {requirements}')


def download_file(url, dest):
    if dest.exists():
        print(f"File already exists, skipping: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def setup_sadtalker():
    checkpoints = [
        # SadTalker checkpoints
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
            "external/SadTalker/checkpoints/mapping_00109-model.pth.tar",
        ),
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
            "checkpoints/mapping_00229-model.pth.tar",
        ),
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
            "external/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors",
        ),
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
            "external/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors",
        ),
        # GFPGAN weights
        (
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
            "external/SadTalker/gfpgan/weights/alignment_WFLW_4HG.pth",
        ),
        (
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "external/SadTalker/gfpgan/weights/detection_Resnet50_Final.pth",
        ),
        (
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "external/SadTalker/gfpgan/weights/GFPGANv1.4.pth",
        ),
        (
            "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "external/SadTalker/gfpgan/weights/parsing_parsenet.pth",
        ),
    ]
    for url, dest in checkpoints:
        download_file(url, Path(dest))


def main():
    import time

    start_time = time.time()

    # 0. Verify Python version is 3.10
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print(f"❌ Python 3.10 is required. Detected: {sys.version.split()[0]}")
        sys.exit(1)

    # 1. Initialize submodules only if not already initialized
    gitmodules = Path(".gitmodules")
    if gitmodules.exists():
        print("Updating git submodules...")
        run("git submodule update --init --recursive")
    else:
        print("No submodules to update.")

    # 2. Create virtual environments and install requirements
    create_venv(".venv-imagegen-sd", "requirements-imagegen-sd.txt", torch_cuda=True)
    create_venv(
        ".venv-animate-sadtalker",
        "requirements-animate-sadtalker.txt",
        torch_cuda=False,
    )

    # 3. Download SadTalker checkpoints and GFPGAN weights
    setup_sadtalker()

    # 4. Sanity checks: run CLI commands in respective environments
    print("Running sanity check: animate-image in .venv-animate-sadtalker...")
    animate_python = (
        Path(".venv-animate-sadtalker")
        / ("Scripts" if os.name == "nt" else "bin")
        / "python"
    )
    run(
        f'"{animate_python}" -m lifelike.app animate-image --image-path .\dataset\\faces\\raghu\\1.jpg --script .\\test\script.txt --output output/animated_video.mp4'
    )

    print("Running sanity check: generate-dreamboothset in .venv-imagegen-sd...")
    imagegen_python = (
        Path(".venv-imagegen-sd") / ("Scripts" if os.name == "nt" else "bin") / "python"
    )
    run(
        f'"{imagegen_python}" -m lifelike.app generate-dreamboothset priya-2 --gender female --ethnicity "indian, light skinned" --age adult --seed-base 9769'
    )

    elapsed = time.time() - start_time
    print(
        f"✅ Project initialization complete and sanity checks run. Total time: {elapsed/60:.2f} minutes."
    )


if __name__ == "__main__":
    main()
