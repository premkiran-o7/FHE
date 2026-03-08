# 🧬 FHE Medical AI — Breast Cancer Classifier

A Streamlit app combining **Fully Homomorphic Encryption**, **Federated Learning**, 
and **Gemini AI** for privacy-preserving breast cancer classification.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

| Tab | Description |
|-----|-------------|
| 📊 Overview | Dataset stats + real-world impact for doctors, patients, hospitals, researchers |
| 📈 Training Curves | Live loss & accuracy charts across epochs |
| 🔬 Weight Analysis | 4-panel weight visualization: ranked bar chart, histogram, heatmap, +/- scatter |
| 🔐 FHE Demo | CKKS encryption session log showing encrypt → transmit → decrypt pipeline |
| 🤖 Gemini Agent | AI-powered clinical report + interactive Q&A for any stakeholder |

## Usage

1. Set **epochs**, **learning rate**, and **oversample** in the sidebar
2. Toggle **FHE Encryption Demo** to see CKKS in action
3. Paste your **Gemini API Key** for the AI agent
4. Click **🚀 Train Model**
5. Navigate tabs to explore results

## Getting a Gemini API Key

1. Visit https://aistudio.google.com/app/apikey
2. Create a new API key (free tier available)
3. Paste it in the sidebar

## Architecture

```
Breast Cancer Dataset (sklearn)
        ↓
StandardScaler + RandomOverSampler
        ↓
PyTorch Logistic Regression (31 features)
        ↓  ← FHE: encrypt weights with TenSEAL CKKS
SGD Training (BCELoss)
        ↓
Weight Analysis Graphs (matplotlib)
        ↓
Gemini 2.0 Flash Agent → Clinical Report + Q&A
```
