# 🧠 AI-Driven Customer Insights and Recommendation System  

This project leverages **Machine Learning** to predict customer value based on behavioral metrics like **Recency, Frequency, and Monetary value (RFM model)**.  
It provides a **Flask-based web interface** where users can input customer data and instantly get an AI-powered prediction of the customer’s value or potential.



## Project Overview  

Businesses rely on customer segmentation and value prediction to drive marketing decisions.  
This project uses data science techniques to:  
- Analyze customer purchase patterns  
- Predict potential customer value  
- Help in identifying high-value customers for targeted strategies  



##  Tech Stack  

- **Python 3.9+**  
- **Flask** (for web app)  
- **Pandas, NumPy, Scikit-learn** (for data preprocessing and modeling)  
- **Matplotlib, Seaborn** (for EDA)  
- **Random Forest Regressor** (final model)  
- **HTML / CSS** (for frontend templates)  
- **AWS / Render / Railway** (for deployment)



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

