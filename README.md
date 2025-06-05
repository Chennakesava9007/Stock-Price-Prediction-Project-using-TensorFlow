# 📈 Stock Price Prediction with LSTM

A complete workflow in Python for **exploratory data analysis (EDA)** and **time‑series forecasting** of Apple (`AAPL`) stock closing prices using a **Long Short‑Term Memory (LSTM)** neural network built with **TensorFlow / Keras**.

---

## 🚀 Project Highlights

- Loads the **`all_stocks_5yr.csv`** dataset (Open‑High‑Low‑Close‑Volume for multiple tickers from 2013‑2018).  
- Performs **EDA** with matplotlib: price trends & trading volume for 9 tech companies.  
- Pre‑processes the Apple close‑price series with `MinMaxScaler` and creates 60‑day sliding windows.  
- Builds a **2‑layer LSTM** model plus dense & dropout layers.  
- Trains for 10 epochs and evaluates on the final 5 % of the data.  
- Reports **MSE / RMSE** and plots actual vs. predicted prices.

---

## 🧪 Quick Preview

```
MSE 4.23
RMSE 2.06

Plot window will open showing:
 • Blue  – training close prices
 • Orange – true test prices
 • Green – LSTM predictions
```

*(Your exact error numbers will depend on random weight initialization and TensorFlow version.)*

---

## 🛠 Requirements

| Package        | Tested Version |
| -------------- | -------------- |
| Python         | 3.9 / 3.10     |
| pandas         | 2.x            |
| numpy          | 1.x            |
| matplotlib     | 3.x            |
| seaborn        | 0.13           |
| scikit‑learn   | 1.x            |
| TensorFlow     | 2.12 / 2.15    |

Install everything with:

```bash
pip install -r requirements.txt
```

*(Create `requirements.txt` with the libs above or use your own environment manager.)*

---

## ▶️ How to Run

```bash
# clone the repo
git clone https://github.com/Chennakesava9007/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm

# place dataset
mkdir -p data && mv /path/to/all_stocks_5yr.csv data/

# run the script
python stock_lstm.py
```

The script will:

1. Print the dataframe shape and `.info()`
2. Show three EDA figures
3. Train the LSTM
4. Display MSE / RMSE
5. Plot Train vs. Test vs. Predictions chart

---

## 📊 Understanding the Code

| Section | Key Lines | Purpose |
| ------- | --------- | ------- |
| **Load & clean** | `pd.read_csv`<br>`pd.to_datetime` | Read CSV, parse date column |
| **EDA** | `plt.subplot` loops | Visualise close / open / volume for multiple tickers |
| **Scaling** | `MinMaxScaler` | Normalize close prices to 0‑1 range |
| **Windowing** | `for i in range(60, …)` | Create 60‑day sequences for LSTM |
| **Model** | `keras.models.Sequential()` | Two LSTM layers 64‑units each + dropout |
| **Training** | `model.fit(..., epochs=10)` | Optimiser = Adam, loss = MSE |
| **Evaluation** | `inverse_transform`, error metrics | Compute & print MSE, RMSE |
| **Plot** | `plt.plot` | Compare real vs. predicted prices |

---

## 🔮 Possible Improvements

- **Increase epochs / tune hyperparameters**  
- Use **Bidirectional LSTM** or **GRU** layers  
- Add **technical indicators** as extra features  
- Implement **walk‑forward validation** for more realistic evaluation  
- Create a **dashboard** (Streamlit / Plotly Dash) for interactive exploration

---

## 📜 License

This project is released under the MIT License.

---

## 👤 Author

[Kesava](https://github.com/Chennakesava9007) • Data & AI Enthusiast

---
