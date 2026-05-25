# 📈 Stock Market Prediction Dashboard

A Machine Learning-based Stock Market Prediction Dashboard built using Python, Streamlit, Scikit-Learn, and Yahoo Finance. The application analyzes historical stock data, calculates technical indicators, predicts future stock prices, and generates Buy/Sell/Hold signals through an interactive web interface.

## 🚀 Features

- Real-time stock data from Yahoo Finance
- Support for multiple stocks (Reliance, TCS, Infosys, Apple, Tesla)
- Technical Indicators:
  - RSI (Relative Strength Index)
  - Moving Averages (MA7, MA21, MA50)
  - Bollinger Bands
  - Volatility Analysis
- Machine Learning Models:
  - Linear Regression
  - Random Forest Regressor
- Buy / Sell / Hold Recommendations
- Interactive Plotly Charts
- Model Performance Evaluation
  - MAE
  - RMSE
  - R² Score
- Responsive Streamlit Dashboard

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- Yahoo Finance API (yfinance)

---

## 📊 Machine Learning Workflow

1. Fetch historical stock market data.
2. Generate technical indicators.
3. Create features for machine learning.
4. Train prediction models.
5. Predict the next day's closing price.
6. Generate trading signals based on predictions.
7. Visualize results through an interactive dashboard.

---

## 📁 Project Structure

```
Stock-Market-Prediction/
│
├── app.py
├── data.py
├── features.py
├── model.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/stock-market-prediction-dashboard.git
```

Navigate to the project folder:

```bash
cd stock-market-prediction-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Supported Stocks

- Reliance Industries
- Tata Consultancy Services (TCS)
- Infosys
- Apple
- Tesla

---

## 📸 Dashboard Preview

Add screenshots here after uploading them.

### Main Dashboard

![Dashboard Screenshot](screenshots/dashboard.png)

### Prediction Results

![Prediction Screenshot](screenshots/prediction.png)

---

## 📋 Model Evaluation Metrics

The application evaluates model performance using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help determine the prediction accuracy and reliability of the trained models.

---

## 🔮 Future Enhancements

- LSTM Deep Learning Model
- News Sentiment Analysis
- Portfolio Recommendation System
- Additional Technical Indicators
- Real-time Prediction Alerts
- More Stock Market Support

---

## 👨‍💻 Author

Sahil Lakra

B.Tech Student | Data Science & Machine Learning Enthusiast

