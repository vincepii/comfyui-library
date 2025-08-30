import websocket
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import random
import copy
from typing import List


class ComfyUIClient:
    """
    A client library to interact with a ComfyUI server for image generation.
    """

    # Node IDs from the workflow, used to modify parameters
    _NODE_CHECKPOINT_LOADER = "4"
    _NODE_POSITIVE_PROMPT = "16"
    _NODE_NEGATIVE_PROMPT = "40"
    _NODE_LATENT_IMAGE = "53"
    _NODE_KSAMPLER = "3"
    _NODE_SAVE_IMAGE = "9"

    # The embedded ComfyUI workflow in API format
    _WORKFLOW_TEMPLATE = {
        "3": {
            "inputs": {
                "seed": 0,  # Placeholder, will be replaced
                "steps": 20,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "sgm_uniform",
                "denoise": 1,
                "model": [_NODE_CHECKPOINT_LOADER, 0],
                "positive": [_NODE_POSITIVE_PROMPT, 0],
                "negative": [_NODE_NEGATIVE_PROMPT, 0],
                "latent_image": [_NODE_LATENT_IMAGE, 0],
            },
            "class_type": "KSampler",
        },
        _NODE_CHECKPOINT_LOADER: {
            "inputs": {
                "ckpt_name": "model.safetensors"  # Placeholder, will be replaced
            },
            "class_type": "CheckpointLoaderSimple",
        },
        "8": {
            "inputs": {
                "samples": [_NODE_KSAMPLER, 0],
                "vae": [_NODE_CHECKPOINT_LOADER, 2],
            },
            "class_type": "VAEDecode",
        },
        _NODE_SAVE_IMAGE: {
            "inputs": {"filename_prefix": "ComfyUI_API", "images": ["8", 0]},
            "class_type": "SaveImage",
        },
        _NODE_POSITIVE_PROMPT: {
            "inputs": {
                "text": "",  # Placeholder, will be replaced
                "clip": [_NODE_CHECKPOINT_LOADER, 1],
            },
            "class_type": "CLIPTextEncode",
        },
        _NODE_NEGATIVE_PROMPT: {
            "inputs": {
                "text": "",  # Placeholder, will be replaced
                "clip": [_NODE_CHECKPOINT_LOADER, 1],
            },
            "class_type": "CLIPTextEncode",
        },
        _NODE_LATENT_IMAGE: {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptySD3LatentImage",
        },
    }

    def __init__(self, server_address="127.0.0.1:8188"):
        """
        Initializes the ComfyUI client.

        Args:
            server_address (str): The address (IP:PORT) of the ComfyUI server.
        """
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def _queue_prompt(self, prompt_workflow):
        """Queues a prompt workflow and returns the server response."""
        p = {"prompt": prompt_workflow, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        try:
            response = urllib.request.urlopen(req)
            return json.loads(response.read())
        except urllib.error.URLError as e:
            print(f"Error queuing prompt: {e}")
            return None

    def _get_image(self, filename, subfolder, folder_type):
        """Fetches an image from the ComfyUI server."""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        try:
            with urllib.request.urlopen(
                f"http://{self.server_address}/view?{url_values}"
            ) as response:
                return response.read()
        except urllib.error.URLError as e:
            print(f"Error fetching image: {e}")
            return None

    def _get_history(self, prompt_id):
        """Gets the history for a given prompt ID."""
        try:
            with urllib.request.urlopen(
                f"http://{self.server_address}/history/{prompt_id}"
            ) as response:
                return json.loads(response.read())
        except urllib.error.URLError as e:
            print(f"Error getting history: {e}")
            return None

    def _get_final_images(self, prompt_id):
        """
        Waits for prompt execution and retrieves the final generated images.
        """
        ws = websocket.WebSocket()
        try:
            ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
            print("Waiting for prompt to finish execution...")

            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "executing":
                        data = message.get("data", {})
                        if (
                            data.get("node") is None
                            and data.get("prompt_id") == prompt_id
                        ):
                            break
        finally:
            ws.close()

        print("Execution finished. Retrieving final images...")
        history = self._get_history(prompt_id)
        if not history or prompt_id not in history:
            print("Could not retrieve execution history.")
            return []

        images = []
        for node_id in history[prompt_id].get("outputs", {}):
            node_output = history[prompt_id]["outputs"][node_id]
            if "images" in node_output:
                for image_data in node_output["images"]:
                    if image_data["type"] == "output":
                        image_bytes = self._get_image(
                            image_data["filename"],
                            image_data["subfolder"],
                            image_data["type"],
                        )
                        if image_bytes:
                            images.append(Image.open(io.BytesIO(image_bytes)))
        return images

    def generate_images(
        self,
        positive_prompt: str,
        negative_prompt: str,
        model_name: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = None,
        steps: int = 20,
        cfg: float = 4.0,
        sampler_name: str = "euler",
        scheduler: str = "sgm_uniform",
    ) -> List[Image.Image]:
        """
        Generates images and returns them as a list of Pillow Image objects.
        Args:
            positive_prompt (str): The positive text prompt.
            negative_prompt (str): The negative text prompt.
            model_name (str): The filename of the checkpoint model (e.g., "model.safetensors").
            width (int): The width of the image.
            height (int): The height of the image.
            seed (int, optional): The seed for generation. If None, a random seed is used.
            steps (int): The number of sampling steps.
            cfg (float): The CFG scale.
            sampler_name (str): The name of the sampler.
            scheduler (str): The name of the scheduler.
        Returns:
            List[Image.Image]: A list of generated Pillow Image objects.
        """
        prompt_workflow = copy.deepcopy(self._WORKFLOW_TEMPLATE)

        prompt_workflow[self._NODE_CHECKPOINT_LOADER]["inputs"][
            "ckpt_name"
        ] = model_name
        prompt_workflow[self._NODE_POSITIVE_PROMPT]["inputs"]["text"] = positive_prompt
        prompt_workflow[self._NODE_NEGATIVE_PROMPT]["inputs"]["text"] = negative_prompt
        prompt_workflow[self._NODE_LATENT_IMAGE]["inputs"]["width"] = width
        prompt_workflow[self._NODE_LATENT_IMAGE]["inputs"]["height"] = height

        ksampler = prompt_workflow[self._NODE_KSAMPLER]["inputs"]
        ksampler["seed"] = (
            seed if seed is not None else random.randint(0, 1_000_000_000)
        )
        ksampler["steps"] = steps
        ksampler["cfg"] = cfg
        ksampler["sampler_name"] = sampler_name
        ksampler["scheduler"] = scheduler

        prompt_data = self._queue_prompt(prompt_workflow)
        if not prompt_data or "prompt_id" not in prompt_data:
            print("Failed to queue prompt.")
            return []

        prompt_id = prompt_data["prompt_id"]
        print(f"Prompt queued successfully with ID: {prompt_id}")

        return self._get_final_images(prompt_id)


# ======================================================================
# EXAMPLE USAGE
# ======================================================================
if __name__ == "__main__":
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
