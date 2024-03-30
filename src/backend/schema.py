from typing import Optional

from pydantic import BaseModel


class Parameters(BaseModel):
    seed: Optional[int] = 42
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
