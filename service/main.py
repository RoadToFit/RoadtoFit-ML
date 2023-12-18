from fastapi import FastAPI, UploadFile
from model import BodyClass, predictBodyClass, recommendActivities
from pydantic import BaseModel

app = FastAPI()

class BodyClassifierResponse(BaseModel):
    success: bool
    message: str
    result: str

class ActivityRecommendationRequest(BaseModel):
    input: str

class ActivityRecommendationResponse(BaseModel):
    success: bool
    message: str
    result: list[int]

@app.post('/model/body-classifier', response_model=BodyClassifierResponse)
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

    return { "success": True, "message": "OK", "result": label }

@app.post('/model/activities-recommendation', response_model=ActivityRecommendationResponse)
async def activities_recommendation(input: ActivityRecommendationRequest):
    result = recommendActivities(input.input)
    return { "success": True, "message": "OK", "result": result }
    