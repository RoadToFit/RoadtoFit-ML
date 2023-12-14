const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const multer = require('multer');
const multerGoogleStorage = require('multer-cloud-storage');
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const fs = require('fs');
const axios = require('axios');

dotenv.config();

const app = express();
const upload = multer({
  storage: multerGoogleStorage.storageEngine({
    acl: 'publicRead',
    destination: 'model_images/',
  }),
});

app.use(cors());

let bodyClassifierModel;

// Load all models
async function loadModel() {
  bodyClassifierModel = await tf.loadLayersModel(
    'file://./models/body-classifier.json'
  );
}

loadModel();

// Preprocess image for body classifier model
async function preprocessImage(imageBuffer) {
  // Decode the image into a tensor
  const imageTensor = tfnode.node.decodeImage(imageBuffer, 3);

  // Resize image to 200x200 size
  const resizedImage = tf.image.resizeBilinear(imageTensor, [200, 200]);

  // Expand image dimension to fit Tensor4D required
  const batchedImage = resizedImage.expandDims(0);

  return batchedImage;
}

// Prediction function for body classifier model
async function predictImage(model, imageBuffer) {
  // Preprocess the image
  const preprocessedImage = await preprocessImage(imageBuffer);
  // Predict the image
  const prediction = model.predict(preprocessedImage);

  // Extract the prediction value
  const value = prediction.dataSync();

  return value;
}

// Body classifier endpoint
app.post('/model/body-classifier', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: 'No image uploaded' });
  }

  try {
    axios
      .get(`https://storage.googleapis.com/roadtofit-bucket/${req.file.path}`, {
        responseType: 'arraybuffer',
      })
      .then(async (response) => {
        const prediction = await predictImage(
          bodyClassifierModel,
          response.data
        );
        return res.status(200).json({ prediction });
      })
      .catch((error) => {
        console.error(error);
        throw new Error();
      });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: 'Internal server error' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
