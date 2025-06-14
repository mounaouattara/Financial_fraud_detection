# Fraud Detection Project

This repository presents a complete data analysis project focused on detecting fraudulent credit card transactions using real-world data and machine learning models.

## Project Structure

```
fraud-detection-project/
├── data/
│   ├── raw/                # Raw dataset (place creditcard.csv here)
│   └── processed/          # Preprocessed datasets (generated by scripts)
├── notebooks/              # Exploratory analysis and result visualization
├── scripts/                # Data preparation and model training scripts
├── models/                 # Saved models (generated by scripts)
├── app/                    # Streamlit dashboard for interactive use
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── fraud_report.pdf        # Summary report
```

## Dataset

The project uses the "Credit Card Fraud Detection" dataset, available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

To use this dataset:

1. Download the `creditcard.csv` file manually from the link above.
2. Place it in the `data/raw/` directory of this repository.

## How to Run

Install the required packages:

```
pip install -r requirements.txt
```

Run the preprocessing and model training scripts:

```
python scripts/preprocess.py
python scripts/train_model.py
```

To launch the interactive dashboard:

```
streamlit run app/streamlit_app.py
```

## Notes
- Processed data and trained models will be generated locally.
- This project was designed to simulate a real-world data analyst workflow.
