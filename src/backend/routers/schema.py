from typing import Optional

from pydantic import BaseModel


class Parameters(BaseModel):
    seed: Optional[int] = 42
    inference_steps:int = 10
    guidance_scale: float = 6

class Task(BaseModel):
    task_id: str
    status: str

class Prediction(Task):
    task_id: str
    status: str
    result: str
