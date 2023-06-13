from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Define the species names
species_names = ['Setosa', 'Versicolor', 'Virginica']

# Define the home route
@app.route('/')
def home():
    return render_template('index.html', species_names=species_names)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the feature values from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Create a NumPy array with the input features
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make the prediction
    prediction = model.predict(input_features)
    predicted_species_encoded = prediction[0]

    # Map the label-encoded species back to their actual names
    predicted_species = species_names[predicted_species_encoded]

    # Render the template with the prediction result
    return render_template('index.html', species_names=species_names, predicted_species=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
