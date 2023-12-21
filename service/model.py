from enum import Enum
import joblib
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import numpy as np
from random import random

class BodyClass(Enum):
    Ectomorph = "ECTOMORPH"
    Endomorph = "ENDOMORPH"
    Mesomorph = "MESOMORPH"

def preprocessImage(imageBuffer: bytes):
    imageTensor = tf.io.decode_image(imageBuffer, 3)
    resizedImage = tf.image.resize(imageTensor, [200,200])
    batchedImage = tf.expand_dims(resizedImage, 0)
    return batchedImage

def predictBodyClass(imageBuffer: bytes) -> np.ndarray:
    image = preprocessImage(imageBuffer)
    model = tf.keras.models.load_model("./model/model1.h5", custom_objects={"CustomRMSprop": RMSprop})

    value = model.predict(image)
    return value

def preprocessActivitiesInput(df: pd.DataFrame, input: str):
    activities = df.drop(["Aktivitas", "kategori", "id"], axis=1)
    encoding = pd.get_dummies(pd.Series(input))
    missing_cols = set(activities.columns) - set(encoding.columns)
    for col in missing_cols:
        encoding[col] = 0

    encoding = encoding[activities.columns]

    return encoding

def recommendActivities(input: str):
    # Read excel file, encode the content, and concat to the main dataframe
    df = pd.read_excel("./model/dataset_workout.xlsx")
    df_encoded = pd.get_dummies(df["kategori"])
    df = pd.concat([df, df_encoded], axis=1)

    # Get preprocessed encoding from the input
    encoding = preprocessActivitiesInput(df, input)

    # Load the model
    model = joblib.load("./model/model3.joblib")
    
    # Do prediction to get the category
    prediction = model.predict(encoding)[0]

    # Post process the output (fetch all the activites with predicted category)
    activities = df[df["kategori"] == prediction][["Aktivitas", "kal/jam", "id"]]
    activities_df = activities.reset_index(drop=True)
    result = activities_df['id'].tolist()

    return result

def calculateBMR(gender: str, height: int, weight: int, age: int) -> float:
    if gender == 'male':
        return 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    elif gender == 'female':
        return 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)

def total_daily_calories(bmr: float, activity: str) -> float:
    if activity == 'sedentary':
        return bmr * 1.2
    elif activity == 'lightly active':
        return bmr * 1.375
    elif activity == 'moderately active':
        return bmr * 1.55
    elif activity == 'very active':
        return bmr * 1.725
    elif activity == 'extra active':
        return bmr * 1.9
    
def calculate_weight_loss_plan_calories(normal_calories: float, plan: str) -> float:
    if plan == 'maintain weight':
        return normal_calories
    elif plan == 'mild weight loss':
        return normal_calories * (0.9 + random() * 0.05)
    elif plan == 'weight loss':
        return normal_calories * (0.8 + random() * 0.1)
    elif plan == 'extreme weight loss':
        return normal_calories * (0.7 + random() * 0.1)
    
def calculate_meal_calories(total_calories: float, num_meals: int):
    if num_meals == 3:
        breakfast_percentage = 0.30
        lunch_percentage = 0.35
        dinner_percentage = 0.25
    elif num_meals == 4:
        breakfast_percentage = 0.25
        morning_snack_percentage = 0.05
        lunch_percentage = 0.35
        dinner_percentage = 0.25
    elif num_meals == 5:
        breakfast_percentage = 0.25
        morning_snack_percentage = 0.05
        lunch_percentage = 0.35
        afternoon_snack_percentage = 0.05
        dinner_percentage = 0.15
    else:
        raise ValueError("Invalid number of meals. Please choose 3, 4, or 5.")

    # Calculate calories for each meal based on percentages
    breakfast_calories = total_calories * breakfast_percentage
    lunch_calories = total_calories * lunch_percentage
    dinner_calories = total_calories * dinner_percentage

    # Create and return a dictionary with the calculated calories for each meal
    meal_calories = {
        'breakfast': breakfast_calories,
        'lunch': lunch_calories,
        'dinner': dinner_calories,
    }

    if num_meals >= 4:
        morning_snack_calories = total_calories * morning_snack_percentage
        meal_calories['morning_snack'] = morning_snack_calories

    if num_meals == 5:
        afternoon_snack_calories = total_calories * afternoon_snack_percentage
        meal_calories['afternoon_snack'] = afternoon_snack_calories

    return meal_calories

def calculate_macronutrient(calories: float, bodyType: str) -> dict[str, float]:
    macronutrients = {'carbohydrates': 0.35, 'protein': 0.35, 'fat': 0.3}

    if bodyType == 'endomorph':
        macronutrients = {'carbohydrates': 0.25, 'protein': 0.4, 'fat': 0.35}
    elif bodyType == 'meshomorp':
        macronutrients = {'carbohydrates': 0.35, 'protein': 0.35, 'fat': 0.3}
    elif bodyType == 'ectomorph':
        macronutrients = {'carbohydrates': 0.4, 'protein': 0.3, 'fat': 0.3}

    # Calculate macronutrients based on percentages
    macronutrient_intake = {
        nutrient: calories * percentage / (4 if nutrient != 'fat' else 9)
        for nutrient, percentage in macronutrients.items()
    }

    return macronutrient_intake

# Fungsi untuk membaca model dan scaler
def load_knn_model(category):
    model_filename = f'./model/{category}_knn_model.joblib'
    scaler_filename = f'./model/{category}_scaler.joblib'

    knn_model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    return knn_model, scaler

# Fungsi untuk melakukan prediksi
def recommendFood(user_input, knn_model, scaler, category):
    df = pd.read_csv('./model/foods_menu.csv')
    # Standarisasi input pengguna
    user_input_scaled = scaler.transform(np.array([user_input]))

    # Mendapatkan indeks makanan terdekat
    _, indices = knn_model.kneighbors(user_input_scaled)

    # Menampilkan rekomendasi makanan
    recommendations = df[df['category'] == category].iloc[indices[0][:3]]
    return recommendations

def postprocessFoodRecommendation(data):
    parsed_data_array = []
    for key, value in data["menu"].items():
        parsed_data_array.append({
            "id": key,
            "menu": value,
            "calories": data["calories"][key],
            "protein": data["protein"][key],
            "fat": data["fat"][key],
            "carbo": data["carbo"][key],
            "image": data["image"][key],
            "category": data["category"][key]
        })

    return parsed_data_array