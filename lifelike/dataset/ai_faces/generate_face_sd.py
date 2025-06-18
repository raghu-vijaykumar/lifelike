from diffusers import StableDiffusionPipeline
import torch

# 1. Load the Stable Diffusion pipeline (first time will download weights)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,  # Disable safety checker
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define your prompt for a face
prompt = (
    "potrait photo of a young woman, realistic, looking at camera, neutral background"
)

# 3. Generate the image
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# 4. Save the image
image.save("face_sd.png")
print("Face image saved as face_sd.png")
