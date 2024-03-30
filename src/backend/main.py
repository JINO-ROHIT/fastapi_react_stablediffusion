from contextlib import asynccontextmanager

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer

from routers import routers


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
sd_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting the models")
    sd_models = {
            "vae": AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(torch_device),
            "tokenizer": CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
            "text_encoder": CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(torch_device),
            "unet": UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(torch_device),
            "scheduler": LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        }
    yield
    sd_models.clear()

app = FastAPI(
    title = "Stable diffusion with FastAPI",
    description = "React and FastAPI",
    version = '0.1',
    lifespan = lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
)

app.include_router(routers.router, prefix="", tags=[""])

