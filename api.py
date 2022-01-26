import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from utilsHandle import *

import os

app = FastAPI(title='ID PTIT OCR by Khong Tung')

# By using @app.get("/") you are allowing the GET method to work for the / endpoint.

dir_path = os.path.dirname(os.path.realpath(__file__))
tmpPath = os.path.join(dir_path, 'tmp')
if os.path.exists(tmpPath):
    shutil.rmtree(tmpPath)
if not os.path.exists(tmpPath):
    os.mkdir(tmpPath)


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Author: Tung Khong Manh. Now head over to " \
           "/docs. "


@app.get("/getCard")
def getCard():
    file_path = get_path().path_Card
    if os.path.exists(file_path):
        return FileResponse(file_path)


@app.get("/getAvatar")
def getAvatar():
    file_path = get_path().path_avatar
    if os.path.exists(file_path):
        return FileResponse(file_path)


@app.post("/upload")
async def uploadImg(fileUpload: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    filename = fileUpload.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    file_location = f"tmp/{fileUpload.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(fileUpload.file.read())
    print(f"info: file {fileUpload.filename} saved at {file_location}")

    res = predict(file_location)

    return {
        "result": res,
    }


# Allows the server to be run in this interactive environment
# nest_asyncio.apply()

# Host depends on the setup you selected (docker or virtual env)
host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

# Spin up the server!
uvicorn.run(app, host=host, port=8000)
