# Shallow Foundation Safety Factor Analysis

![Earthquake](https://img.shields.io/badge/Domain-Earthquake%20Engineering-orange)
![ML](https://img.shields.io/badge/ML-Regression%20Analysis-blue)
![Python](https://img.shields.io/badge/Language-Python-green)
![Models](https://img.shields.io/badge/Algorithms-8%20Models-red)

## Overview

This repository contains a comprehensive collection of regression analysis files developed to predict the safety factor of shallow foundations under earthquake conditions. Multiple machine learning algorithms were explored and compared to create an accurate prediction model.

## Purpose

The primary goal of this research is to develop an artificial intelligence model that can accurately predict the safety factor (SF) of shallow foundations during seismic events. This predictive capability is crucial for earthquake-resistant construction and structural engineering.

## Dataset

The analysis uses a dataset with the following parameters:

### Input Variables
- **PGA**: Peak Ground Acceleration (g)
- **Length/Width**: Foundation dimensions (m)
- **Depth**: Foundation depth (m)
- **Soil Density**: Unit weight of soil (kN/m³)
- **Load**: Applied structural load (kN)

### Output Variable
- **SF**: Safety Factor (dimensionless)

## Algorithms Explored

This repository contains implementation and evaluation of multiple regression algorithms:

1. **MLR**: Multiple Linear Regression
2. **DT**: Decision Tree Regression
3. **KNN**: K-Nearest Neighbors Regression
4. **SVR**: Support Vector Regression
5. **XGBoost**: Extreme Gradient Boosting
6. **TabNet**: High-performance interpretable tabular learning
7. **MLP**: Multi-Layer Perceptron (Neural Network)
8. **KAN**: Kolmogorov-Arnold Networks (latest implementation)

## Repository Structure

```
/
├── data/
│   ├── Dataset_Qult_and_SF.csv
│   ├── test_120225.csv
│   └── train_120225.csv
├── model/
│   ├── Data Cleaning.ipynb
│   ├── Evaluation.ipynb
│   ├── Modeling : KAN (previous).ipynb
│   ├── Modeling : KAN (recent).ipynb
│   └── Modeling : MLR DT KNN SVR XGBoost TabNet MLP.ipynb
├── LICENSE
└── README.md
```

## Workflow

The analysis follows a structured workflow:

1. **Data Cleaning** (`Data Cleaning.ipynb`): 
   - Data preprocessing
   - Handling missing values
   - Feature engineering
   - Data normalization/standardization

2. **Model Development**:
   - Traditional ML models (`Modeling : MLR DT KNN SVR XGBoost TabNet MLP.ipynb`)
   - KAN implementation - previous version (`Modeling : KAN (previous).ipynb`)
   - KAN implementation - optimized version (`Modeling : KAN (recent).ipynb`)

3. **Model Evaluation** (`Evaluation.ipynb`):
   - Performance metrics calculation
   - Model comparison
   - Error analysis
   - Cross-validation results

## Key Findings

The KAN (Kolmogorov-Arnold Networks) algorithm demonstrated superior performance for this specific task, leading to its selection for the final implementation in the SAFOUND web application. This repository documents the progression and comparison that led to this conclusion.

## Data Description

- `Dataset_Qult_and_SF.csv`: Complete dataset with input parameters and safety factors
- `train_120225.csv`: Training dataset created on February 12, 2025
- `test_120225.csv`: Test dataset for model validation

## Usage

To explore these notebooks:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shallow-foundation-analysis.git
   cd shallow-foundation-analysis
   ```

2. Ensure you have the required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost pytorch pytorch-tabnet tensorflow pykan
   ```

3. Open the Jupyter notebooks to explore the analysis:
   ```bash
   jupyter notebook
   ```

## Results

The comparative analysis revealed:

- Traditional methods like MLR provided baseline performance but failed to capture complex relationships
- Tree-based models (DT, XGBoost) showed improved performance due to their ability to capture non-linear patterns
- Neural network approaches (MLP, TabNet) demonstrated good results but required more extensive hyperparameter tuning
- KAN emerged as the superior approach, effectively capturing the complex relationships between seismic parameters and foundation safety factors

## Implementation

The best-performing model from this analysis has been implemented in a web application called SAFOUND, which allows for real-time prediction of safety factors based on input parameters. The web application is available in a separate repository.

## Future Work

Potential areas for improvement include:
- Incorporating additional soil parameters
- Exploring ensemble methods combining multiple model predictions
- Developing region-specific models that account for local geological conditions
- Extending the model to handle different foundation types

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.

## Acknowledgments

- The research community for developing and sharing advanced regression algorithms
- Contributors who helped in data collection and validation
- Testing and verification partners who helped ensure model accuracy
