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

def recommendCalories(moderat: str, berat: str, kal_jam: str):
    input = { "berat": berat, "kal/jam": kal_jam, "moderat": moderat }
    series = pd.Series(data=input, index=["berat", "kal/jam", "moderat"])
    print(series)
    encoding = pd.get_dummies(series)
    model = joblib.load('./model/model3.joblib')
    
    value = model.predict(encoding)
    return value