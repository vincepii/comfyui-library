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

# --- Call the generation function with your desired parameters ---
success = client.generate_image(
    positive_prompt="A stunningly beautiful cinematic shot of a futuristic neon-lit city street at night, wet pavement reflecting the glowing signs, flying vehicles in the sky, high detail, 8k",
    negative_prompt="(worst quality, low quality, normal quality), blurry, ugly, disfigured, watermark, text, signature, plain background",
    model_name=MODEL_NAME,
    output_path=OUTPUT_FILE_PATH,
    width=1024,
    height=1024,
    steps=25,
    cfg=4.5,
    seed=42 # Use a fixed seed for reproducibility
)

if not success:
    print("Image generation failed.")
```