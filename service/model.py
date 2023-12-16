from enum import Enum
import joblib
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import numpy as np

class BodyClass(Enum):
    Ectomorph = "Ectomorph"
    Endomorph = "Endomorph"
    Mesomorph = "Mesomorph"

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

def preprocessCaloriesInput(df: pd.DataFrame, input: str):
    calories = df.drop(['Aktivitas', 'kategori'], axis=1)
    encoding = pd.get_dummies(pd.Series(input))
    missing_cols = set(calories.columns) - set(encoding.columns)
    for col in missing_cols:
        encoding[col] = 0
    
    encoding = encoding[calories.columns]

    return encoding

def recommendCalories(input: str):
    # Read excel file, encode the content, and concat to the main dataframe
    df = pd.read_excel("./model/dataset_workout.xlsx")
    df_encoded = pd.get_dummies(df['kategori'])
    df = pd.concat([df, df_encoded], axis=1)

    # Get preprocessed encoding from the input
    encoding = preprocessCaloriesInput(df, input)

    # Load the model
    model = joblib.load("./model/model3.joblib")
    
    # Do prediction to get the category
    prediction = model.predict(encoding)[0]

    # Post process the output (fetch all the activites with predicted category)
    activities = df[df["kategori"] == prediction][["Aktivitas", "kal/jam"]]
    activities_df = activities.reset_index(drop=True)
    result = activities_df['Aktivitas'].tolist()

    return result