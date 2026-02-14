Trading on Trends
Advanced Stock Sentiment Analysis System
An intelligent, end-to-end system that combines social media sentiment, financial market data, and technical analysis to predict stock price movements. Built using Python, Flask, and advanced machine learning models, this project enables real-time analysis, predictive modeling, and interactive visualizations.

Key Features
API Integrations
Reddit API (PRAW) Collects sentiment data from financial subreddits Method: fetch_reddit_data()

Yahoo Finance API (yfinance) Fetches OHLCV data and historical stock prices Method: fetch_stock_data()

Flask API Provides web endpoints for real-time analysis Route: /analyze

Data Preprocessing and Transformation
Text Cleaning Removes URLs, special characters, and normalizes whitespace Method: clean_text()

Data Merging Aligns stock and sentiment data based on dates Method: merge_stock_and_sentiment()

Missing Value Handling Uses forward/backward fill for stock data and zero-fill for sentiment

Feature Engineering Includes technical indicators like RSI, MACD, Bollinger Bands Adds lag features, rolling means, and interaction terms Method: add_technical_indicators()

Machine Learning Models
Classification (Stock Price Direction - Up/Down)
Logistic Regression (L2 Regularization)

Random Forest Classifier

XGBoost Classifier Methods: train_models(), _evaluate_logistic_model()

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

Includes confusion matrix visualization

Regression (Stock Return %)
Ridge Regression

Gradient Boosting Regressor

XGBoost Regressor Methods: train_models(), _evaluate_linear_model()

Evaluation Metrics: RMSE, MAE, R² Score, Directional Accuracy

Advanced Capabilities
Interactive Web Application
Built using Flask for API interaction
Real-time visualization using Plotly
Dashboard includes price charts, volume, sentiment, and indicators
Enhanced Sentiment Analysis
Custom financial lexicon applied with VADER
Aggregates sentiment from multiple subreddits
Time-aware features like sentiment momentum and rolling sentiment averages
Training Pipeline Optimization
Random Forest-based feature importance selection
Time-series cross-validation for temporal modeling
SMOTE for addressing class imbalance
Automatic comparison of multiple models
Robust Error Handling
Handles missing values and infinite values
API error fallbacks during data collection
Input validation to ensure safe plotting and analysis
Performance Monitoring
Stores model metrics in JSON
Visual comparison of feature importance across models
Backtesting for historical strategy validation
Modular and Scalable Design
Clean separation of data, model, and visualization components
Easy to extend with new data sources or algorithms
Configurable analysis parameters
Project Structure
stock_sentiment_analysis/
├── data/
│   ├── raw/              # Original API data
│   ├── processed/        # Cleaned and merged datasets
│   └── final/            # Analysis-ready data
├── models/               # Trained model artifacts
├── visualizations/       # Generated charts and plots
├── app.py                # Flask web application
├── StockSentimentAnalyzer.py  # Main analysis class
└── requirements.txt      # Python dependencies
Future Enhancements
Real-time streaming from Reddit and stock tickers
Deep learning models (LSTM, Transformer-based architectures)
Multi-asset portfolio analysis
Risk management strategies and position sizing
Deployment to cloud platforms (AWS, GCP, Azure)
Responsive web front-end with mobile support
