# CO₂ Emissions Prediction using Elastic Net Regression 🚗💨

This project predicts CO₂ emissions (in g/km) for vehicles based on
engine and fuel consumption data using **Elastic Net Regression**.

## 📁 Project Structure

    CO2_Emissions_Predictor/
    │
    ├── model/
    │   ├── co2_model.pkl
    │   └── scaler.pkl
    │
    ├── templates/
    │   └── index.html
    │
    ├── app.py
    └── model_train.py

## 🧠 Model: `model_train.py`

-   Uses `ElasticNet` from scikit-learn
-   Scales features using `StandardScaler`
-   Evaluates model with R² score and MSE
-   Saves model and scaler with `pickle`

## 🌐 Flask App: `app.py`

-   Accepts user input for 6 features:
    -   Engine Size (L)
    -   Cylinders
    -   Fuel Consumption City (L/100 km)
    -   Fuel Consumption Hwy (L/100 km)
    -   Fuel Consumption Comb (L/100 km)
    -   Fuel Consumption Comb (mpg)
-   Returns predicted CO₂ emission

## 🎨 Frontend: `templates/index.html`

-   Simple HTML form for user inputs
-   Displays prediction or error message

## Images

[!ElasticRegression](Images\image.png)

## ▶️ How to Run

1.  Train the model: `bash     python model_train.py`

2.  Run the Flask app: `bash     python app.py`

3.  Open browser and go to: `http://127.0.0.1:5000/`

## 📦 Requirements

Install dependencies:

    pip install pandas numpy scikit-learn flask

## 📊 Sample Output

-   R² Score: \~0.90
-   MSE: \~341.57

------------------------------------------------------------------------

Made with ❤️ for ML projects.
