import pickle
import torch
import numpy as np
from flask import Flask, render_template, request
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Define the Model class again (like in your training script)
class Model(nn.Module):
    def __init__(self, in_features=8, h1=8, h2=9, h3=8, out_features=2):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

# Initialize the Flask app
app = Flask(__name__)

# Load the model weights
model = Model()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Load the saved StandardScaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Create a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Create a route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form (assuming 8 features for the diabetes model)
        features = [float(request.form.get(f'feature{i}')) for i in range(1, 9)]
        input_data = np.array(features).reshape(1, -1)  # Reshape for a single sample

        # Apply the same scaling transformation that was used during training
        input_data_scaled = scaler.transform(input_data)

        # Convert the scaled data to a PyTorch tensor and make predictions
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor)

        # Extract the predicted class (for binary classification)
        predicted_class = torch.argmax(prediction, dim=1).item()
        if predicted_class==1:
            prediction_text="Positive"
        else:
            prediction_text="Negative"
        # Return the prediction result
    except Exception as e:
        prediction_text = f"Error: {str(e)}"  # Handle error and display it

    # Return the prediction result
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
