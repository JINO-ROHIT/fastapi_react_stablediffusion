from typing import Any, Dict, List

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, PNDMScheduler, UNet2DConditionModel
from fastapi import APIRouter, Depends
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer

from src.backend.main import sd_models
from src.backend.routers.schema import Prediction, Task


router = APIRouter()


@router.get("/")
def welcome() -> Dict[str, str]:
    return {"hello": "there"}

@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}

@router.get("/predict")
async def predict():
    print(sd_models)
    result = sd_models['vae']
    return {"result": result}

# @router.post("/predict")
# def predict(data: List[List[float]]):
#     crud.create_entry(data)
#     return {"success": "data received for prediction!"}

# @router.post("/predict_batch")
# def predict_batch():
#     current = time.time()
#     logger.info("Batch job triggered")
#     crud.predict_all()
#     logger.info(f"Batch Job took {time.time() - current} seconds")
#     return {"success": "batch prediction done!"}