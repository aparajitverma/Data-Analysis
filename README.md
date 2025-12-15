# 🏦 German Credit Risk Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

A comprehensive machine learning analysis of the German Credit Risk dataset to predict loan default risk using various classification models.

## 📊 Overview

This project explores credit risk assessment using the UCI German Credit dataset. The analysis includes:
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and preprocessing
- Multiple machine learning models comparison
- Model evaluation and interpretation
- Interactive prediction interface

## 🎯 Key Findings

- **Best Model**: Logistic Regression with 73.2% accuracy and 0.763 ROC-AUC
- **Top Predictors**: Credit amount, duration, account status, housing
- **Feature Engineering**: Log-transforms and missing value indicators improved performance by ~3%
- **Business Impact**: Enables data-driven credit decisions with interpretable results

## 📁 Project Structure

```
german-credit-risk/
│
├── german_credit_data.csv          # Dataset (UCI German Credit)
├── german_credit_eda.ipynb         # Main analysis notebook
├── requirements.txt                # Python dependencies
├── linkedin_blog_german_credit copy.md  # Blog post summary
├── output/                         # Generated plots and results
│   └── images/                     # Visualization images
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook/Lab

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/german-credit-risk.git
   cd german-credit-risk
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Running the Analysis
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `german_credit_eda.ipynb`

3. Run all cells sequentially to reproduce the analysis

### Key Sections
- **Data Loading & Preprocessing**: Handle missing values, encoding
- **EDA**: Univariate/bivariate analysis, correlations
- **Modeling**: Train and compare 7 ML models
- **Evaluation**: Confusion matrices, ROC curves, feature importance
- **Interactive UI**: Web-based prediction interface

## 📊 Dataset

**Source**: UCI Machine Learning Repository (Statlog German Credit Data)

**Size**: 1,000 observations, 20 features + target

**Target**: Credit risk (good/bad)

**Features**:
- Demographics: Age, Sex, Job
- Financial: Credit amount, Duration
- Accounts: Checking/Saving accounts status
- Other: Housing, Purpose, etc.

**Challenge**: 30% class imbalance, missing values in account features

## 🤖 Models Compared

| Model | Accuracy | ROC-AUC | Recall (Bad) |
|-------|----------|---------|--------------|
| **Logistic Regression** | 73.2% | 0.763 | 69.3% |
| XGBoost | 72.8% | 0.738 | 48.0% |
| LightGBM | 74.4% | 0.731 | 41.3% |
| Random Forest | 70.8% | 0.689 | 45.3% |
| Gradient Boosting | 70.4% | 0.721 | 40.0% |
| Neural Network | 72.0% | 0.717 | 46.7% |
| Deep Neural Network | 72.0% | 0.717 | 46.7% |

## 🔍 Key Insights

### Risk Factors
- **High Credit Amounts**: Associated with higher default rates
- **Long Durations**: Extended loan terms increase risk
- **Account Status**: "Little" balances are strong risk indicators
- **Housing**: Renters show higher bad rates than owners

### Model Performance
- Logistic Regression provides best balance of accuracy and interpretability
- Feature engineering (log-transforms, missing indicators) improved performance
- ROC-AUC ceiling around 0.75 suggests dataset limitations

## 🛠️ Technologies Used

- **Python**: Core language
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: ML models and preprocessing
- **Matplotlib/Seaborn**: Visualization
- **XGBoost/LightGBM**: Advanced ensemble methods
- **Jupyter**: Interactive development
- **IPyWidgets**: Interactive UI

## 📈 Results & Visualizations

The analysis generates various plots saved in `output/images/`:
- Distribution plots
- Correlation heatmaps
- Confusion matrices
- ROC curves
- Feature importance charts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn, XGBoost, and other open-source libraries
- Inspiration from various credit risk analysis projects

## 📞 Contact

For questions or suggestions:
- Open an issue on GitHub
- Connect on LinkedIn

---

**#MachineLearning #CreditRisk #DataScience #Finance #Python #EDA**