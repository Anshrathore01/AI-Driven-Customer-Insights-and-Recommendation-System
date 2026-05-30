---
title: AI Driven Customer Insights And Recommendation System
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🧠 AI-Driven Customer Insights and Recommendation System  

This project leverages **Machine Learning** to predict customer value based on behavioral metrics like **Recency, Frequency, and Monetary value (RFM model)**.  
It provides a **Flask-based web interface** where users can input customer data and instantly get an AI-powered prediction of the customer’s value or potential.

## Project Overview  

Businesses rely on customer segmentation and value prediction to drive marketing decisions.  
This project uses data science techniques to:  
- Analyze customer purchase patterns  
- Predict potential customer value  
- Help in identifying high-value customers for targeted strategies  

## Tech Stack  

- **Python 3.9+** - **Flask** (for web app)  
- **Pandas, NumPy, Scikit-learn** (for data preprocessing and modeling)  
- **Matplotlib, Seaborn** (for EDA)  
- **Random Forest Regressor** (final model)  
- **HTML / CSS** (for frontend templates)  
- **HUGGING SPACES** (for deployment)

## 📂 Project Structure  

AI-Driven-Customer-Insights-and-Recommendation-System/
│
├── app.py                             # Flask web application entry point
├── requirements.txt                   # All project dependencies
│
├── src/                               # Main source code folder
│   ├── components/                    # Data ingestion, transformation, model training modules
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/                      # End-to-end pipelines
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── utils.py                       # Utility functions (save/load models, evaluate)
│   ├── logger.py                      # Custom logging configuration
│   └── exception.py                   # Custom exception handling
│
├── templates/                         # Frontend HTML files
│   ├── index.html
│   └── home.html
│
├── artifacts/                         # Auto-generated data & model storage
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
│
└── README.md                          # Project documentation

## 🚀 How to Run the App Locally (Beginner Friendly)

If you are a beginner and just want to run the application natively on your computer without Docker, follow these simple steps:

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd AI-Driven-Customer-Insights-and-Recommendation-System
   ```

2. **Create a Virtual Environment** (Recommended):
   This keeps your project dependencies isolated.
   ```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```

4. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:
   ```bash
   python app.py
   ```

6. **Open the App in your Browser**:
   Once the server starts, open your web browser and go to:
   👉 **http://127.0.0.1:7860**

   > 💡 **Note:** The first time you run the app, it might take **1-2 minutes** to load all the Machine Learning models (like XGBoost, CatBoost) into memory. Please wait until you see the `* Running on http://127.0.0.1:7860` message in your terminal before opening the link.


## 🐳 Deploy on Hugging Face Spaces (Docker)

The repository now ships with a production-ready `Dockerfile`, making it easy to deploy on [Hugging Face Spaces](https://huggingface.co/spaces) using the **Docker** runtime.

1. Create a new Space and choose **Docker** as the SDK.
2. Push this repository to the Space (or connect it as a Git submodule/mirror).
3. The Space will automatically build the provided `Dockerfile`. The app listens on `PORT=7860` as required by Spaces.

### Run the same image locally

```bash
docker build -t customer-insights:latest .
docker run -p 7860:7860 customer-insights:latest