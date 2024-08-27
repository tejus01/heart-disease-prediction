# heart-disease-prediction
Based on the contents of the "Heart Disease Prediction with Neural Networks" notebook, here's a draft for a README file:

---

# Heart Disease Detection with Neural Networks

This project aims to predict the likelihood of heart disease in patients using neural networks. The project leverages various machine learning techniques and is implemented in a Jupyter Notebook.

## Project Overview

The main steps in this project include:

1. **Data Preparation**:
   - Loading and cleaning the dataset.
   - Feature selection and engineering.
   - Splitting the dataset into training and test sets.

2. **Model Building**:
   - Construction of two types of neural network models:
     - A categorical model for predicting multiple classes.
     - A binary model for predicting whether a patient has heart disease or not.
   - Model architecture includes layers such as input, dense, and output layers, with activation functions like ReLU and softmax.

3. **Model Training**:
   - The models are trained using the dataset.
   - Hyperparameter tuning is done to optimize the model's performance.

4. **Evaluation**:
   - The performance of the models is evaluated using accuracy, precision, recall, and F1-score metrics.
   - A classification report is generated for both models.

5. **Prediction**:
   - The trained models are used to make predictions on new data.

## Files in the Repository

- `Heart Disease Prediction with Neural Networks.ipynb`: The main Jupyter notebook containing the code for the entire process.
- `model_binary.h5`: The trained binary classification model saved in HDF5 format (if the notebook saves it).
- `model_categorical.h5`: The trained categorical classification model saved in HDF5 format (if the notebook saves it).

## How to Use

1. **Setup**:
   - Ensure you have Python installed along with the necessary libraries. The required packages are listed below.

2. **Running the Notebook**:
   - Open the `Heart Disease Prediction with Neural Networks.ipynb` file in Jupyter Notebook or Jupyter Lab.
   - Run all cells to see the data processing, model training, and prediction steps.

3. **Making Predictions**:
   - Use the trained models to predict the likelihood of heart disease in patients based on the features provided.

4. **Exporting Results**:
   - The trained models can be saved and later used for deployment or further analysis.

## Required Packages

To run the notebook, you will need to install the following packages:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `scikit-learn`: For model evaluation and metrics.
- `tensorflow` or `keras`: For building and training the neural network models.
- `matplotlib`: For plotting and visualization.

You can install these packages using `pip`:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

## Example Usage

Here's an example of how you might use the trained binary model to predict heart disease:

```python
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model_binary.h5')

# Example input data: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
input_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

# Predict
prediction = model.predict(input_data)
print("Likelihood of heart disease:", prediction[0][0])
```

## Author

[Tejus Sharma]
