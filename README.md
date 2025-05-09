# Loan Default Prediction with BERT & Topic Modeling
This project predicts the likelihood of a borrower defaulting on a loan by combining structured financial data with unstructured borrower narratives. It uses traditional machine learning models enhanced with BERT-derived risk scores, SHAP for model explainability, and BERTopic for thematic topic modeling.

## Project Overview
- **Goal**: Accurately predict loan default and understand the features — both numerical and textual — driving risk.
- **Dataset**: LendingClub loan data with financial attributes and borrower-provided loan descriptions.(Link:https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv )
- **Approach**:
  - Traditional models (Logistic Regression, Random Forest, XGBoost)
  - BERT-based risk scoring from borrower narratives
  - SHAP for interpretability
  - BERTopic to extract borrower intent themes
--
## Key Features
- **Fine-tuned BERT** model to convert borrower descriptions into a default risk score.
- **Traditional ML models** using structured + BERT features.
- **SHAP explainability** to show what drives each prediction.
- **BERTopic** to extract common narrative clusters and link them to risk.
- Evaluation with accuracy, F1-score, ROC-AUC, threshold tuning.


## Project Summary

- Built a fine-tuned BERT model to generate borrower `bert_risk_score`
- Trained and evaluated Logistic Regression, Random Forest, and XGBoost
- Applied SHAP for global and local feature-level interpretability
- Used BERTopic to cluster borrower intent and assess topic-level risk
- Best performance: **Logistic Regression (F1: 0.7153, ROC-AUC: 0.8647)**

## Research Questions

- Can borrower narratives meaningfully improve credit risk prediction?
- How do narrative themes correlate with loan default probability?
- Which features — structured or unstructured — are most predictive?
- How can we make our risk models more explainable?

## Key Takeaways

1. BERT-derived `bert_risk_score` was the **most impactful feature** across all models.
2. Logistic Regression outperformed tree-based models with simpler feature interaction.
3. SHAP force plots provided case-level transparency into individual predictions.
4. BERTopic revealed **high-risk narrative clusters** (e.g., business loans, medical bills).
5. Adding topic IDs as features did not improve ML performance but enhanced interpretability.

## How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction

# 2. Install dependencies
pip install -r requirements.txt
