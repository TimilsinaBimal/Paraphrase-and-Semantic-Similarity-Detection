from fastapi import FastAPI, Request
import json
from tensorflow.keras import models
from data.preprocessing import prepare_data
from utils.config import BINARY_VOCAB_PATH
from models.predict import predict_binary, predict_binary_single


app = FastAPI()


@app.post("/api/result/")
async def result(request: Request):
    request_body = await request.body()
    request_body = request_body.decode('utf8').replace("'", '"')
    request_body = json.loads(request_body)
    sentence1 = request_body['sentence1']
    sentence2 = request_body['sentence2']
    data1, data2 = prepare_data(
        [sentence1],
        [sentence2],
        BINARY_VOCAB_PATH,
        training=False
    )
    model = models.load_model('binary_loss_model.h5')
    pred = predict_binary_single(model, data1, data2)
    response = {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "Success": "Hello"
    }
    # print(request_body)
    return response


@app.get("/")
def index():
    return {"data": "OK"}
