import io
from contextlib import asynccontextmanager
from typing import Dict, Optional

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from utils.fastapi_globals import g


torch_device = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting the models")

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(torch_device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(torch_device)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(torch_device)
    scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    g.set_default("vae", vae)
    g.set_default("tokenizer", tokenizer)
    g.set_default("text_encoder", text_encoder)
    g.set_default("unet", unet)
    g.set_default("scheduler", scheduler)

    yield
    del vae, tokenizer, text_encoder, unet, scheduler
    g.cleanup()

app = FastAPI(
    title = "Stable diffusion with FastAPI",
    description = "React and FastAPI",
    version = '0.1',
    lifespan = lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def welcome() -> Dict[str, str]:
    return {"hello": "there"}

@app.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}

@app.get("/predict")
async def predict(prompt: str, batch_size:int = 1, num_inference_steps:Optional[int] = 100, guidance_scale:Optional[float] = 7.5):
    text_input = g.tokenizer(prompt, padding="max_length", max_length=g.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = g.text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = g.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = g.text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn(
            (batch_size, g.unet.in_channels, 512 // 8, 512 // 8),
            generator = torch.manual_seed(42),
            )
    latents = latents.to(torch_device)
    g.scheduler.set_timesteps(num_inference_steps)
    latents = latents * g.scheduler.init_noise_sigma

    for t in tqdm(g.scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = g.scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = g.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = g.scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = g.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    memory_stream = io.BytesIO()
    pil_images[0].save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")

