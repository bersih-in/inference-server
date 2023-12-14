import numpy as np
from keras.models import load_model
from PIL import Image
from fastapi import FastAPI, BackgroundTasks, status, UploadFile, File, Depends
from io import BytesIO
from fastapi.exceptions import HTTPException
import asyncio
import httpx
from dotenv import load_dotenv
from functools import lru_cache
from typing_extensions import Annotated

from .config import Settings
from .models.inference import InferenceModel, InferenceAsyncModel

load_dotenv()

MODEL_PATH = "model/mymodel.h5"
model = load_model(MODEL_PATH)
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = model.input_shape[1:]

app = FastAPI()

# Global variable to track download status
download_in_progress = False

@lru_cache()
def get_settings():
    return Settings()


@app.get("/")
async def read_root():
    return {"Status": "Inference Server"}

@app.get("/settings")
async def read_settings(settings: Annotated[Settings, Depends(get_settings)]):
    return {
        "app_name": settings.app_name,
        "backend_endpoint": settings.BACKEND_ENDPOINT,
    }


@app.post("/inference-link")
async def inference_link(body: InferenceModel):
    imageUrl = body.imageUrl

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(imageUrl)
            image_data = response.content

        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_resize = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        image_array = np.asarray(image_resize)
        image_array = image_array.reshape(
            1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)
        image_array = image_array.astype('float32') / 255.0

        result = model.predict(image_array)

        return {"success": True, "result": result[0].tolist()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/inference-file")
async def inference_file(file: UploadFile):
    try:
        image_data = await file.read()

        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_resize = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        image_array = np.asarray(image_resize)
        image_array = image_array.reshape(
            1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)
        image_array = image_array.astype('float32') / 255.0

        result = model.predict(image_array)

        return {"success": True, "result": result[0].tolist()}
    except Exception as e:
        return {"success": False, "error": str(e)}
    

@app.put("/inference-async-link")
async def inference_file_async(body: InferenceAsyncModel, background_tasks: BackgroundTasks):
    imageUrl = body.imageUrl
    submissionId = body.submissionId

    background_tasks.add_task(inference_link_task_async, imageUrl, submissionId)
    return {"success": True, "message": "Image is being processed"}

@app.get("/pull-model", include_in_schema=False)
async def pull_model(background_tasks: BackgroundTasks):
    if download_in_progress:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model is already being downloaded",
        )
    background_tasks.add_task(download_model)
    return {"success": True, "message": "Model is downloading"}


async def download_model():
    global download_in_progress

    if download_in_progress:
        return
    download_in_progress = True

    # model download code goes here
    await asyncio.sleep(10)

    download_in_progress = False


async def inference_link_task_async(imageUrl: str, submissionId: int):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(imageUrl)
            image_data = response.content

        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_resize = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        image_array = np.asarray(image_resize)
        image_array = image_array.reshape(
            1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)
        image_array = image_array.astype('float32') / 255.0

        result = model.predict(image_array)

        if result[0][0] > 0.5:
            result = "VERIFIED"
        else:
            result = "REJECTED_BY_ML"

        # push result to backend
        async with httpx.AsyncClient() as client:
            response = await client.put(
                get_settings().BACKEND_ENDPOINT,
                json={
                    "submissionId": submissionId,
                    "status": result
                }
            )
            print(response.text)
    except Exception as e:
        print(e)