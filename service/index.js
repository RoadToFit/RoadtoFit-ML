import express, { json } from 'express';
import cors from 'cors';
import { loadLayersModel, image } from '@tensorflow/tfjs';
import { node } from '@tensorflow/tfjs-node';
import multer from 'multer';
import { readFileSync } from 'fs';

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(json());
app.use(cors());

let bodyClassifierModel;

// Load all models
async function loadModel() {
  bodyClassifierModel = await loadLayersModel(
    'file://./models/body-classifier.json'
  );
}

loadModel();

// Preprocess image for body classifier model
async function preprocessImage(imageBuffer) {
  // Decode the image into a tensor
  const imageTensor = node.decodeImage(imageBuffer, 3);

  // Resize image to 200x200 size
  const resizedImage = image.resizeBilinear(imageTensor, [200, 200]);

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
    const imageBuffer = readFileSync(req.file.path);
    const prediction = await predictImage(bodyClassifierModel, imageBuffer);

    return res.status(200).json({ prediction });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: 'Internal server error' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
