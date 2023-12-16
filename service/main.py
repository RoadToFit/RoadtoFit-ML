from fastapi import FastAPI, Request, UploadFile
from model import BodyClass, predictBodyClass, recommendCalories

app = FastAPI()

@app.post('/model/body-classifier')
async def body_classifier(image: UploadFile):
    label = ""
    buffer = image.file.read()
    result = predictBodyClass(buffer)
    result_list = result.tolist()

    for index, value in enumerate(result_list[0]):
        if value == 1.0:
            if index == 0:
                label = BodyClass.Ectomorph
            elif index == 1:
                label = BodyClass.Endomorph
            else:
                label = BodyClass.Mesomorph

    return { "result": label }

@app.post('/model/calories-recommendation')
async def calories_recommendation(request: Request):
    input = await request.json()
    result = recommendCalories(input["moderat"], input["berat"], input["kal_jam"])
    return result
    