from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Prediction page

@app.route('/predict', methods=['POST'])
def predict_data():
    try:
        # Get data from form
        recency = float(request.form.get('Recency'))
        frequency = float(request.form.get('Frequency'))
        monetary = float(request.form.get('Monetary'))

        # Create DataFrame for model input
        data = CustomData(
            Recency=recency,
            Frequency=frequency,
            Monetary=monetary
        )
        pred_df = data.get_data_as_dataframe()

        # Predicting using the pipeline
        pipeline = PredictPipeline()
        prediction = pipeline.predict(pred_df)[0]

        # Render results
        return render_template('home.html',
                               prediction_text=f"Predicted Customer Value: {prediction:.2f}")

    except Exception as e:
        return render_template('home.html',
                               prediction_text=f"Error: {e}")



# Running the app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
