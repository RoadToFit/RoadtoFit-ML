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
    model = tf.keras.models.load_model('./model/model1.h5', custom_objects={'CustomRMSprop': RMSprop})

    value = model.predict(image)
    return value

def preprocessCalories(input: str):
    encoding = pd.get_dummies(pd.Series(input))

    # Lengkapi data dengan kolom dummy yang mungkin hilang
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # Sesuaikan urutan kolom agar sesuai dengan model yang telah dilatih
    input_data = input_data[X.columns]

def recommendCalories(input: str):
    
    model = joblib.load('./model/model3.joblib')
    
    value = model.predict(encoding)
    return value