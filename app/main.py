import numpy as np
from keras.models import load_model
from PIL import Image
from fastapi import FastAPI, BackgroundTasks, status, UploadFile, File
from io import BytesIO
from fastapi.exceptions import HTTPException
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "model/mymodel.h5"
model = load_model(MODEL_PATH)
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = model.input_shape[1:]

app = FastAPI()

# Global variable to track download status
download_in_progress = False


@app.get("/")
async def read_root():
    return "Inference Server"


@app.post("/inference")
async def inference(imageUrl: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(imageUrl)
            image_data = response.content

        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_resize = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        image_array = np.asarray(image_resize)
        print(image_array.shape)
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
        print(image_array.shape)
        image_array = image_array.reshape(
            1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)
        image_array = image_array.astype('float32') / 255.0

        result = model.predict(image_array)

        return {"success": True, "result": result[0].tolist()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/pull-model")
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
