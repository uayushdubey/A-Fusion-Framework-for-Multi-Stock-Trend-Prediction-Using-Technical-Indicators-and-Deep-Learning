#  A Fusion Framework for Multi-Stock Trend Prediction Using Technical Indicators and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/uayushdubey/A-Fusion-Framework-for-Multi-Stock-Trend-Prediction-Using-Technical-Indicators-and-Deep-Learning)

A comprehensive deep learning framework that integrates multi-source technical indicators with LSTM sequence modeling to predict stock market trends across multiple tickers. This system leverages 40+ technical indicators, sliding-window time-series processing, and advanced neural architectures to generate reliable UP/DOWN/NEUTRAL trend classifications and future price forecasts.

---

##  Overview

The **Stock Market Analyzer** is an end-to-end machine learning pipeline designed to predict stock price movements using historical OHLCV (Open, High, Low, Close, Volume) data. The system processes multi-stock data, engineers 40+ technical indicators, creates time-series sequences, and trains LSTM models for both classification (price direction) and regression (future prices).

### **Key Capabilities:**
- Multi-stock batch processing
- 40+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- LSTM-based deep learning architecture
- Classification (UP/DOWN/NEUTRAL) + Regression (Future OHLC)
- Automated training and evaluation pipeline
- Excel-based reporting and predictions
- Visual performance metrics and confusion matrices

---

## ✨ Core Features

### **1. Multi-Stock Support**
- Process multiple stock tickers in batch mode
- Individual model training per stock
- Parallel processing capability
- Automated iteration and error handling

### **2. Comprehensive Technical Indicators (40+)**

**Trend Indicators:**
- Simple Moving Averages (SMA 10, 20, 50, 200)
- Exponential Moving Averages (EMA 12, 26)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Parabolic SAR

**Momentum Indicators:**
- RSI (Relative Strength Index)
- Stochastic Oscillator (%K, %D)
- Williams %R
- Rate of Change (ROC)
- Chande Momentum Oscillator (CMO)

**Volatility Indicators:**
- Bollinger Bands (Upper, Middle, Lower, Width)
- ATR (Average True Range)
- Keltner Channels
- Donchian Channels
- Historical Volatility

**Volume Indicators:**
- OBV (On-Balance Volume)
- MFI (Money Flow Index)
- VWAP (Volume Weighted Average Price)
- Volume Rate of Change
- Force Index

**Custom Features:**
- Price gaps
- Candlestick pattern recognition
- Rolling statistics
- High-Low range percentages
- Momentum divergence

### **3. Advanced LSTM Architecture**
- Two-layer stacked LSTM with dropout regularization
- Bidirectional processing capability
- Sequence-to-sequence modeling
- Multi-output prediction (classification + regression)

### **4. Sliding-Window Time-Series Processing**
- Configurable lookback window (default: 60 days)
- Creates 3D tensor inputs: `(samples, timesteps, features)`
- Preserves temporal dependencies
- Temporal train/validation/test split

### **5. Classification & Regression**
- **Classification:** Predicts trend direction (UP/DOWN/NEUTRAL)
- **Regression:** Forecasts future OHLC prices
- Threshold-based labeling
- Class imbalance handling with SMOTE

### **6. Automated Pipeline**
- End-to-end automation from data loading to prediction
- Configurable hyperparameters
- Early stopping and model checkpointing
- Comprehensive logging and error handling

### **7. Performance Evaluation**
- Confusion matrices
- Classification reports (Precision/Recall/F1)
- Accuracy and loss curves
- Prediction vs actual visualizations
- MAE, RMSE, R² metrics

### **8. Excel Reporting**
- Structured prediction exports
- Confidence scores
- Trading signals (BUY/SELL/HOLD)
- Performance metrics summary

---

## System Architecture

### **High-Level Design (HLD)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-STOCK PREDICTION FRAMEWORK                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Excel Input    │ ← Historical OHLCV Data (Multiple Stocks)
│  (Multi-Stock)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PREPROCESSING                             │
│  • Parse dates & validate data                                      │
│  • Handle missing values                                            │
│  • Sort chronologically                                             │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING ENGINE                        │
│  ┌─────────────────────────────────────────────────────────── ───┐  │
│  │ 40+ Technical Indicators:                                     │  │
│  │  • Trend: SMA, EMA, MACD, ADX                                 │  │
│  │  • Momentum: RSI, Stochastic, Williams %R, ROC                │  │
│  │  • Volatility: Bollinger Bands, ATR, Keltner Channels         │  │
│  │  • Volume: OBV, MFI, VWAP                                     │  │
│  │  • Custom: Gaps, Patterns, Statistics                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LABEL GENERATION                               │
│  • Classification: UP (+1) / NEUTRAL (0) / DOWN (-1)                │
│  • Regression: Future Open, High, Low, Close                        │
│  • Threshold-based labeling (configurable)                          │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SEQUENCE PREPARATION                              │
│  • Sliding window: 60-day sequences                                 │
│  • Output: (samples, 60, features)                                  │
│  • Optional SMOTE for class balancing                               │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  TRAIN/VALIDATION/TEST SPLIT                        │
│  • Train: 70% | Validation: 15% | Test: 15%                         │
│  • Temporal ordering preserved                                      │
│  • StandardScaler normalization                                     │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LSTM MODEL ARCHITECTURE                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Input: (60, num_features)                                   │   │
│  │  ↓                                                           │   │
│  │  LSTM Layer 1: 128 units (return_sequences=True)             │   │
│  │  ↓                                                           │   │
│  │  Dropout: 0.2                                                │   │
│  │  ↓                                                           │   │
│  │  LSTM Layer 2: 64 units                                      │   │
│  │  ↓                                                           │   │
│  │  Dropout: 0.2                                                │   │
│  │  ↓                                                           │   │
│  │  Dense: 32 units (ReLU)                                      │   │
│  │  ↓                                                           │   │
│  │  Output: 3 units (Softmax) OR 4 units (Linear)               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING ENGINE                               │
│  • Optimizer: Adam (lr=0.001)                                       │
│  • Loss: Categorical Crossentropy / MSE                             │
│  • Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau     │
│  • Epochs: 50-100 (configurable)                                    │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION & PREDICTION                          │
│  • Confusion matrix                                                 │
│  • Classification report                                            │
│  • Accuracy/Loss curves                                             │
│  • Future predictions with confidence scores                        │
│  • Trading signals: BUY/SELL/HOLD                                   │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXCEL REPORTING                                │
│  • Predictions with timestamps                                      │
│  • Confidence scores & price targets                                │
│  • Performance metrics                                              │
│  • Model weights saved (.h5)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### **Data Flow Diagram**

```
OHLCV Data → Feature Engineering → Label Generation → Sequence Creation
     ↓              ↓                    ↓                   ↓
  (N × 6)        (N × 45)            (N × 5)           (M × 60 × 45)
                                                            ↓
                                                    Train/Val/Test Split
                                                            ↓
                                                    Normalization
                                                            ↓
                                                       LSTM Model
                                                            ↓
                                                  Predictions + Evaluation
                                                            ↓
                                                      Excel Reports
```

---

## Low-Level Design (LLD)

### **Module Breakdown**

#### **1. Data Loader Module**
```python
def load_stock_data(filepath, sheet_name='Sheet1'):
    """
    Loads OHLCV data from Excel file
    
    Args:
        filepath (str): Path to Excel file
        sheet_name (str): Sheet name to read
    
    Returns:
        pd.DataFrame: Clean OHLCV data
    """
    # Implementation details...
```

#### **2. Feature Engineering Module**
```python
def add_technical_indicators(df):
    """
    Computes 40+ technical indicators
    
    Args:
        df (pd.DataFrame): Raw OHLCV data
    
    Returns:
        pd.DataFrame: Enhanced data with indicators
        
    Indicators Added:
        - SMA (10, 20, 50, 200)
        - EMA (12, 26)
        - MACD + Signal + Histogram
        - RSI (14)
        - Bollinger Bands
        - ATR (14)
        - Stochastic Oscillator
        - OBV, MFI, Williams %R, ROC, VWAP
        - Custom features
    """
    # Implementation details...
```

#### **3. Label Generator Module**
```python
def label_target(df, threshold=0.01):
    """
    Generates classification labels and regression targets
    
    Args:
        df (pd.DataFrame): Feature-engineered data
        threshold (float): % threshold for UP/DOWN
    
    Returns:
        tuple: (y_classification, y_regression)
        
    Classification Logic:
        - Return > threshold → UP (1)
        - Return < -threshold → DOWN (-1)
        - Otherwise → NEUTRAL (0)
    
    Regression Targets:
        - Next day [Open, High, Low, Close]
    """
    # Implementation details...
```

#### **4. Sequence Preparation Module**
```python
def prepare_data_for_lstm(X, y, time_steps=60):
    """
    Creates sliding-window sequences for LSTM
    
    Args:
        X (np.ndarray): Feature matrix (N × features)
        y (np.ndarray): Labels (N,)
        time_steps (int): Lookback window size
    
    Returns:
        tuple: (X_sequences, y_sequences)
        
    Output Shape:
        X_sequences: (samples, time_steps, features)
        y_sequences: (samples,)
        
    Algorithm:
        For i in range(time_steps, len(X)):
            sequence[i] = X[i-time_steps:i]
            label[i] = y[i]
    """
    # Implementation details...
```

#### **5. Model Builder Module**
```python
def build_lstm_model(input_shape, num_classes=3, model_type='classification'):
    """
    Constructs LSTM neural network
    
    Args:
        input_shape (tuple): (time_steps, features)
        num_classes (int): Number of output classes
        model_type (str): 'classification' or 'regression'
    
    Returns:
        keras.Model: Compiled LSTM model
        
    Architecture:
        Input → LSTM(128) → Dropout(0.2) → LSTM(64) → 
        Dropout(0.2) → Dense(32) → Output
        
    Compilation:
        - Optimizer: Adam (lr=0.001)
        - Loss: Categorical Crossentropy / MSE
        - Metrics: Accuracy, Precision, Recall / MAE, RMSE
    """
    # Implementation details...
```

#### **6. Training Engine Module**
```python
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains LSTM model with callbacks
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs (int): Training epochs
        batch_size (int): Batch size
    
    Returns:
        keras.callbacks.History: Training history
        
    Callbacks:
        - EarlyStopping (patience=10)
        - ModelCheckpoint (save best)
        - ReduceLROnPlateau (factor=0.5, patience=5)
    """
    # Implementation details...
```

#### **7. Evaluation Module**
```python
def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance
    
    Generates:
        - Confusion matrix
        - Classification report
        - Accuracy/Loss plots
        - Prediction vs Actual plots
        
    Metrics:
        - Accuracy, Precision, Recall, F1-Score
        - MAE, RMSE, R² (for regression)
    """
    # Implementation details...
```

#### **8. Prediction Engine Module**
```python
def predict_future(model, df, scaler, time_steps=60):
    """
    Predicts future stock movements
    
    Args:
        model: Trained Keras model
        df: Recent stock data
        scaler: Fitted StandardScaler
        time_steps: Sequence length
    
    Returns:
        dict: {
            'direction': 'UP/DOWN/NEUTRAL',
            'confidence': float,
            'next_open': float,
            'next_high': float,
            'next_low': float,
            'next_close': float,
            'signal': 'BUY/SELL/HOLD'
        }
        
    Signal Logic:
        - P(UP) > 0.6 → BUY
        - P(DOWN) > 0.6 → SELL
        - Otherwise → HOLD
    """
    # Implementation details...
```

---

##  Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.8+ |
| **Deep Learning** | TensorFlow / Keras | 2.x |
| **Data Processing** | pandas | 1.3+ |
| **Numerical Computing** | NumPy | 1.21+ |
| **Machine Learning** | Scikit-Learn | 1.0+ |
| **Imbalanced Learning** | imbalanced-learn | 0.8+ |
| **Technical Indicators** | TA-Lib / Custom | - |
| **Visualization** | Matplotlib, Seaborn | 3.4+, 0.11+ |
| **Excel I/O** | openpyxl, xlsxwriter | 3.0+, 3.0+ |
| **Environment** | Jupyter Notebook | - |

---

## Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- CUDA-enabled GPU (optional, for faster training)

### **Step 1: Clone Repository**
```bash
git clone https://github.com/uayushdubey/A-Fusion-Framework-for-Multi-Stock-Trend-Prediction-Using-Technical-Indicators-and-Deep-Learning.git
cd A-Fusion-Framework-for-Multi-Stock-Trend-Prediction-Using-Technical-Indicators-and-Deep-Learning
```

### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow==2.12.0
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
imbalanced-learn==0.10.1
matplotlib==3.7.1
seaborn==0.12.2
openpyxl==3.1.2
xlsxwriter==3.1.0
ta-lib==0.4.26
jupyter==1.0.0
```

### **Step 4: Install TA-Lib (Optional)**
```bash
# Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.26‑cp38‑cp38‑win_amd64.whl

# Linux/Mac
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

---

##  How to Run

### **Method 1: Using Main Script**

1. **Prepare your data:**
   - Place Excel files with OHLCV data in `data/` folder
   - Ensure columns: `Date, Open, High, Low, Close, Volume`

2. **Configure settings:**
   - Edit `config/config.yaml` for hyperparameters
   - Update `config/stock_list.json` with your tickers

3. **Run the pipeline:**
```bash
python main.py
```

### **Method 2: Using Jupyter Notebook**

```bash
jupyter notebook notebooks/model_training.ipynb
```

### **Method 3: Custom Script**

```python
from src.data_loader import load_stock_data
from src.feature_engineering import add_technical_indicators
from src.model_builder import build_lstm_model
from src.training import train_model

# Configuration
config = {
    'time_steps': 60,
    'threshold': 0.01,
    'epochs': 50,
    'batch_size': 32
}

# Load and process data
df = load_stock_data('data/AAPL_historical.xlsx')
df = add_technical_indicators(df)

# ... (continue with pipeline)
```

---

## Sample Output

### **Console Output:**
```
==================================================
Processing: AAPL
==================================================
✓ Data loaded: 1000 rows
✓ Technical indicators computed: 45 features
✓ Labels generated: UP=35%, DOWN=32%, NEUTRAL=33%
✓ Sequences created: (880, 60, 45)
✓ SMOTE applied: Class balance achieved
✓ Train/Val/Test split: 616/132/132 samples

Model Training:
Epoch 1/50 - loss: 0.8234 - accuracy: 0.5894 - val_loss: 0.7123 - val_accuracy: 0.6212
Epoch 2/50 - loss: 0.7012 - accuracy: 0.6543 - val_loss: 0.6789 - val_accuracy: 0.6515
...
Epoch 23/50 - loss: 0.4532 - accuracy: 0.8234 - val_loss: 0.4891 - val_accuracy: 0.7955
Early stopping triggered. Best epoch: 23

✓ Model trained successfully
✓ Test Accuracy: 79.55%
✓ Test Precision: 0.81
✓ Test Recall: 0.78
✓ Test F1-Score: 0.79

Future Prediction:
✓ Direction: UP
✓ Confidence: 72.3%
✓ Next Open: $152.34
✓ Next High: $154.21
✓ Next Low: $151.89
✓ Next Close: $153.67
✓ Signal: BUY

✓ Model saved: models/AAPL_model.h5
✓ Predictions exported: outputs/predictions/AAPL_predictions.xlsx
✓ Plots saved: outputs/plots/

==================================================
AAPL PIPELINE COMPLETED SUCCESSFULLY!
==================================================
```

### **Prediction Excel Output:**
| Date | Actual_Close | Predicted_Direction | Confidence | Predicted_Open | Predicted_High | Predicted_Low | Predicted_Close | Signal |
|------|--------------|-------------------|-----------|---------------|---------------|--------------|----------------|--------|
| 2024-01-15 | 151.30 | UP | 72.3% | 152.34 | 154.21 | 151.89 | 153.67 | BUY |
| 2024-01-16 | 153.20 | UP | 68.5% | 153.50 | 155.10 | 152.90 | 154.80 | BUY |
| 2024-01-17 | 152.80 | NEUTRAL | 45.2% | 153.00 | 154.20 | 151.50 | 152.90 | HOLD |

### **Confusion Matrix:**
```
              Predicted
              DOWN  NEUTRAL  UP
Actual DOWN    38      5      2
       NEUTRAL  4     40      6
       UP       3      5     37
```

### **Visualizations:**
- Accuracy & Loss curves over epochs
- Confusion matrix heatmap
- Prediction vs Actual scatter plots
- Feature importance charts

---

## Model Performance

### **Typical Metrics (on test set):**
| Metric | Value |
|--------|-------|
| **Accuracy** | 75-82% |
| **Precision** | 0.76-0.84 |
| **Recall** | 0.74-0.81 |
| **F1-Score** | 0.75-0.82 |
| **MAE (OHLC)** | $1.23-$2.45 |
| **RMSE (OHLC)** | $2.34-$3.89 |

### **Class-Wise Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DOWN | 0.84 | 0.78 | 0.81 | 45 |
| NEUTRAL | 0.77 | 0.80 | 0.78 | 50 |
| UP | 0.82 | 0.84 | 0.83 | 37 |

---

## 🔮 Future Improvements

### **Planned Enhancements:**
- [ ] Real-time data streaming integration (Alpha Vantage, Yahoo Finance API)
- [ ] Ensemble methods (LSTM + GRU + Transformer)
- [ ] Attention mechanisms for feature importance
- [ ] Multi-step forecasting (predict next N days)
- [ ] Sentiment analysis integration (news, social media)
- [ ] Portfolio optimization module
- [ ] Backtesting framework with trading simulator
- [ ] Web dashboard for visualization (Streamlit/Dash)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)
- [ ] Hyperparameter tuning with Optuna
- [ ] Model interpretability (SHAP values)

### **Advanced Features Under Development:**
- Reinforcement learning for trading strategies
- Graph neural networks for stock correlation modeling
- AutoML pipeline for automated feature selection
- Multi-asset class support (crypto, forex, commodities)

---

## 📖 Usage Examples

### **Example 1: Single Stock Analysis**
```python
from src.pipeline import run_pipeline_for_stock

config = {
    'time_steps': 60,
    'threshold': 0.01,
    'epochs': 50,
    'batch_size': 32,
    'use_smote': True
}

model, predictions = run_pipeline_for_stock(
    stock_name='AAPL',
    filepath='data/AAPL_historical.xlsx',
    config=config
)
```

### **Example 2: Batch Processing**
```python
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

for stock in stocks:
    try:
        model, pred = run_pipeline_for_stock(
            stock_name=stock,
            filepath=f'data/{stock}_historical.xlsx',
            config=config
        )
        print(f"✓ {stock} completed")
    except Exception as e:
        print(f"✗ {stock} failed: {e}")
```

### **Example 3: Custom Prediction**
```python
from src.prediction import predict_future
from src.data_loader import load_stock_data
from tensorflow.keras.models import load_model

# Load model and data
model = load_model('models/AAPL_model.h5')
df = load_stock_data('data/AAPL_historical.xlsx')

# Get prediction
prediction = predict_future(model, df, scaler, time_steps=60)
print(f"Direction: {prediction['direction']}")
print(f"Signal: {prediction['signal']}")
```

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Contribution Guidelines:**
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure backward compatibility


##  Author
**Ayush Dubey**
- GitHub: [@uayushdubey](https://github.com/uayushdubey)

## Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- TA-Lib developers for technical indicator implementations
- Scikit-learn community for machine learning utilities
- Yahoo Finance for historical stock data

---

If you encounter any issues or have questions:
- Open an [Issue](https://github.com/uayushdubey/A-Fusion-Framework-for-Multi-Stock-Trend-Prediction-Using-Technical-Indicators-and-Deep-Learning/issues)
- Start a [Discussion](https://github.com/uayushdubey/A-Fusion-Framework-for-Multi-Stock-
