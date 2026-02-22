# Kunskapskontroll_mnist
School project: handwritten digit classification using the MNIST dataset.

## Repository structure

- **Inlämning_josefin_carlson.ipynb**
Model training, evaluation, and selection of the final model and hyperparameters.

- **Streamlit_app/**   
All files needed to run the Streamlit application.

- **streamlit_josefin_carlson.ipynb**  
  Trains the final model on the full MNIST dataset for deployment in the Streamlit app (`app_numbers_MNIST.py`).

## Requirements

Install dependencies:
    pip install streamlit numpy scikit-learn joblib

## How to run the app
1. Go to the `Streamlit_app` folder. 
2. Run: 
    streamlit run app_numbers_MNIST.py
