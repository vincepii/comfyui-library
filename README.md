# ComfyUI Library

A Library to query ComfyUI.

## Usage

```bash
pip install git+https://github.com/vincepii/comfyui-library.git
```

## Full Example

```python
from comfy_client import ComfyUIClient
# --- Configuration ---
# 🚨 UPDATE THIS WITH YOUR SERVER'S ADDRESS
COMFYUI_SERVER_ADDRESS = "192.168.178.200:8188"

# 🚨 UPDATE THIS WITH THE NAME OF YOUR MODEL FILE
MODEL_NAME = "playground-v2.5-1024px-aesthetic.fp16.safetensors"

OUTPUT_FILE_PATH = "generated_image.png"

# --- Create a client instance ---
client = ComfyUIClient(COMFYUI_SERVER_ADDRESS)

# --- Call the generation function to get a list of image objects ---
generated_images = client.generate_images(
    positive_prompt="A stunningly beautiful cinematic shot of a futuristic neon-lit city street at night, wet pavement reflecting the glowing signs, flying vehicles in the sky, high detail, 4k",
    negative_prompt="(worst quality, low quality, normal quality), blurry, ugly, disfigured, watermark, text, signature, plain background",
    model_name=MODEL_NAME,
    steps=25,
    cfg=4.5,
)

# --- Process the returned images ---
if generated_images:
    print(f"✅ Generation successful! Received {len(generated_images)} image(s).")

    # Save the first image to a file
    try:
        generated_images[0].save(OUTPUT_FILE_PATH)
        print(f"Image saved to {OUTPUT_FILE_PATH}")
    except IOError as e:
        print(f"Error saving image: {e}")

else:
    print("❌ Image generation failed. No images were returned.")
```
