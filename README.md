# Credit Risk Modeling for BNPL Service

## Project Overview

This project builds an end-to-end credit scoring system for **Bati Bank's** new Buy-Now-Pay-Later (BNPL) service in partnership with an eCommerce platform. The system transforms customer transaction data into a predictive credit risk score, enabling automated loan approval decisions.

**Key Deliverables:**
- Credit risk model with interpretable outputs
- Containerized REST API for real-time scoring
- Automated CI/CD pipeline for testing and deployment
- MLflow experiment tracking and model registry

---

## Credit Scoring Business Understanding

### How Basel II Influences Model Interpretability and Documentation

The **Basel II Capital Accord** sets regulatory expectations for financial institutions' risk measurement practices. For credit scoring models, Basel II mandates:

1. **Transparency Requirements**
   - Models must be understandable to risk officers, auditors, and regulators
   - Every modeling decision must be justified and documented
   - The "use test" requires that model outputs are actually used in decision-making

2. **Documentation Standards**
   - Complete model development lifecycle documentation
   - Variable selection rationale and performance metrics
   - Validation results on out-of-time samples
   - Regular monitoring and recalibration procedures

3. **Interpretability as Compliance**
   - Black-box models face regulatory scrutiny unless accompanied by explainability tools (SHAP, LIME)
   - Weight of Evidence (WoE) transformations are preferred because they create monotonic relationships
   - Logistic regression with WoE is a Basel II-friendly standard due to clear coefficient interpretation

**Impact on this project:** We must prioritize interpretable features, document all transformations, and provide clear justification for our proxy target variable.

### Proxy Variable Necessity and Business Risks

**Why a proxy is necessary:**
The raw transaction dataset contains no historical default labels. Banks typically need 12-24 months of loan performance data to observe defaults. Since this is a new BNPL service, we must infer credit risk from behavioral patterns (RFM analysis) as a proxy for default probability.

**Business risks of proxy-based prediction:**

| Risk Category | Description | Mitigation Strategy |
|--------------|-------------|---------------------|
| **False Positives** | Low-risk customers labeled as high-risk → lost revenue, poor customer experience | Conservative cluster assignment; manual review for borderline cases |
| **False Negatives** | High-risk customers labeled as low-risk → unexpected defaults, financial loss | Higher risk tolerance for initial period; continuous model validation |
| **Concept Drift** | Customer behavior patterns change over time | Monthly model retraining; monitoring distribution shifts |
| **Regulatory Risk** | Proxy may have disparate impact on protected groups | Fairness audits; documenting proxy limitations explicitly |
| **Validation Challenge** | Cannot directly validate against true defaults | A/B testing with small exposure; collect ground truth data over time |

**Risk mitigation commitment:** This proxy is an **initial approximation** that will be validated against actual default data as it becomes available. The bank should collect ground truth labels for 6-12 months and recalibrate the model.

### Trade-offs: Interpretable vs. High-Performance Models

In regulated financial contexts, the choice between simple interpretable models and complex high-performance models involves significant trade-offs:

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|-----------|---------------------------|---------------------------------------|
| **Interpretability** | ✅ High - coefficients directly show impact | ❌ Low - black box ensemble |
| **Regulatory Acceptance** | ✅ Standard for Basel II compliance | ⚠️ Requires SHAP/LIME explanations |
| **Predictive Performance** | ⭐⭐ Good baseline | ⭐⭐⭐⭐ Usually superior (5-15% better AUC) |
| **Training Speed** | ✅ Very fast | ⚠️ Slower, requires tuning |
| **Debugging Ease** | ✅ Easy to identify problematic variables | ❌ Difficult to trace decisions |
| **Feature Engineering** | Requires WoE transformation | Handles raw features well |
| **Overfitting Risk** | Low with regularization | Higher, needs careful tuning |
| **Explainability to Customers** | ✅ Clear reasons for denial | ❌ "Algorithm said no" |

**Our Recommendation for Bati Bank:**

Implement a **two-model strategy**:

1. **Primary Model: Logistic Regression with WoE**
   - Used for regulatory reporting and customer explanations
   - Provides Basel II-compliant decisioning
   - Serves as the "official" credit scorecard

2. **Secondary Model: XGBoost with SHAP explanations**
   - Used for edge cases and higher-risk segments
   - Generates feature importance for model improvement insights
   - Provides performance uplift where interpretability is less critical

**Decision Framework:**
- Loan amount < $500 → Logistic Regression only
- Loan amount $500-$2000 → Ensemble (both models must agree)
- Loan amount > $2000 → XGBoost + SHAP explanations for human review

This balanced approach maintains regulatory compliance while capturing performance gains where they matter most.

---

## Project Structure


credit-risk-model/
├── .github/workflows/
│ └── ci.yml # CI/CD pipeline
├── data/
│ ├── raw/ # Raw data (gitignored)
│ └── processed/ # Processed data (gitignored)
├── notebooks/
│ └── eda.ipynb # Exploratory analysis
├── src/
│ ├── init.py
│ ├── data_processing.py # Feature engineering pipeline
│ ├── train.py # Model training with MLflow
│ ├── predict.py # Inference functions
│ └── api/
│ ├── main.py # FastAPI application
│ └── pydantic_models.py # Request/response schemas
├── tests/
│ └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md


---

## Getting Started

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt