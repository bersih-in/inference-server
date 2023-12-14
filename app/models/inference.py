from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

class InferenceModel(BaseModel):
    imageUrl: str

class InferenceAsyncModel(BaseModel):
    imageUrl: str
    submissionId: int