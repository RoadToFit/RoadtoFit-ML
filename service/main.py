from fastapi import FastAPI, UploadFile
from model import BodyClass, predictBodyClass, recommendActivities, calculateBMR, total_daily_calories, calculate_weight_loss_plan_calories, calculate_meal_calories, calculate_macronutrient, load_knn_model, recommendFood, postprocessFoodRecommendation
from pydantic import BaseModel
import json

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

class FoodRecommendationRequest(BaseModel):
    gender: str
    height: int
    weight: int
    age: int
    activity: str
    plan: str
    num_meals: int
    bodyType: str

class FoodRecommendationResponse(BaseModel):
    success: bool
    message: str
    result: list

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
    
@app.post('/model/food-recommendation', response_model=FoodRecommendationResponse)
async def food_recommendation(input: FoodRecommendationRequest):
    bmr = calculateBMR(input.gender, input.height, input.weight, input.age)
    daily_calories = total_daily_calories(bmr, input.activity)
    weight_loss_plan = calculate_weight_loss_plan_calories(daily_calories, input.plan)
    calories_distribution = calculate_meal_calories(weight_loss_plan, input.num_meals)

    meal_category = ["breakfast", "lunch", "dinner", "morning_snack", "afternoon_snack"]
    requested_categories = []

    if (input.num_meals == 3):
        requested_categories = meal_category[0:3]
    elif (input.num_meals == 4):
        requested_categories = meal_category[0:4]
    elif (input.num_meals == 5):
        requested_categories = meal_category[0:5]

    final_result = []

    for category in requested_categories:
        # Get required calories for a meal category
        meal_calories = calories_distribution[category]

        # Calculate the macronutrient based on the required calories and body type
        macronutrient = calculate_macronutrient(meal_calories, input.bodyType)

        # Put the input together
        preprocessedInput = [meal_calories, macronutrient['protein'], macronutrient['fat'], macronutrient['carbohydrates']]

        # Load model and scaler
        knn_model, scaler = load_knn_model(category)

        # Get recommendation prediction
        result = recommendFood(preprocessedInput, knn_model, scaler, category)

        # Process output
        jsonify_result = json.loads(result.to_json())
        parsed_result = postprocessFoodRecommendation(jsonify_result)
        category_dict = {f"{category}": parsed_result}
        final_result.append(category_dict)

    return { "success": True, "message": "OK", "result": final_result }