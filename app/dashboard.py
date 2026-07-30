# import json
# from pathlib import Path

# import pandas as pd
# import streamlit as st

# # ----------------------------
# # Page Configuration
# # ----------------------------
# st.set_page_config(
#     page_title="Credit Risk Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# # ----------------------------
# # Paths
# # ----------------------------
# DATA_PATH = Path("data/processed/final_training_data.csv")
# METRICS_PATH = Path("reports/metrics.json")

# CONFUSION_MATRIX = Path("reports/confusion_matrix.png")
# ROC_CURVE = Path("reports/roc_curve.png")
# FEATURE_IMPORTANCE = Path("reports/feature_importance.png")
# SHAP_SUMMARY = Path("reports/shap_summary.png")
# # ----------------------------
# # Load Data
# # ----------------------------
# @st.cache_data
# def load_data():
#     return pd.read_csv(DATA_PATH)


# @st.cache_data
# def load_metrics():
#     with open(METRICS_PATH) as f:
#         return json.load(f)


# df = load_data()
# metrics = load_metrics()

# # ----------------------------
# # Sidebar
# # ----------------------------
# st.sidebar.title("Navigation")

# page = st.sidebar.radio(
#     "Go to",
#     [
#         "🏠 Home",
#         "📊 Dataset Overview",
#         "📈 Model Performance",
#         "🧠 Explainability",
#         "💼 Business Insights",
#     ],
# )

# # =====================================================
# # HOME
# # =====================================================

# if page == "🏠 Home":

#     st.title("💳 Credit Risk Modeling Dashboard")

#     st.markdown(
#         """
#         ### Business Problem

#         Many customers have limited or no traditional credit history,
#         making loan approval difficult. This project predicts customer
#         credit risk using transaction behavior to support better lending
#         decisions and reduce default risk.
#         """
#     )

#     st.divider()

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Project Information")

#         st.write("**Model:** Random Forest")
#         st.write("**Dataset:** Xente Transaction Data")
#         st.write("**Target:** is_high_risk")

#     with col2:
#         st.subheader("Best Model Performance")

#         st.metric(
#             "ROC-AUC",
#             f"{metrics['roc_auc']:.4f}",
#         )

#         st.metric(
#             "Accuracy",
#             f"{metrics['accuracy']:.4f}",
#         )

#     st.divider()

#     st.subheader("Project Workflow")

#     st.markdown("""
# 1. Data Collection
# 2. Data Preprocessing
# 3. Feature Engineering (RFM)
# 4. Model Training
# 5. Model Evaluation
# 6. Business Insights
# """)

# # =====================================================
# # DATASET PAGE
# # =====================================================

# elif page == "📊 Dataset Overview":

#     st.title("📊 Dataset Overview")

#     col1, col2, col3 = st.columns(3)

#     col1.metric(
#         "Rows",
#         len(df),
#     )

#     col2.metric(
#         "Features",
#         df.shape[1],
#     )

#     col3.metric(
#         "High Risk Customers",
#         int(df["is_high_risk"].sum()),
#     )

#     st.divider()

#     st.subheader("Target Distribution")

#     st.bar_chart(df["is_high_risk"].value_counts())

#     st.divider()

#     st.subheader("Dataset Preview")

#     st.dataframe(df.head())

# # =====================================================
# # MODEL PERFORMANCE
# # =====================================================

# elif page == "📈 Model Performance":

#     st.title("📈 Model Performance")

#     st.write(
#         "The Random Forest model was selected as the best-performing model "
#         "after hyperparameter tuning using GridSearchCV."
#     )

#     st.divider()

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
#     col2.metric("Precision", f"{metrics['precision']:.4f}")
#     col3.metric("Recall", f"{metrics['recall']:.4f}")

#     col4, col5 = st.columns(2)

#     col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
#     col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

#     st.divider()

#     st.subheader("Best Hyperparameters")

#     st.table(
#         {
#             "Parameter": [
#                 "n_estimators",
#                 "max_depth",
#                 "min_samples_split",
#             ],
#             "Value": [
#                 200,
#                 "None",
#                 2,
#             ],
#         }
#     )

#     st.divider()

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Confusion Matrix")

#         if CONFUSION_MATRIX.exists():
#             st.image(str(CONFUSION_MATRIX))
#         else:
#             st.warning("Confusion matrix not found.")

#     with col2:
#         st.subheader("ROC Curve")

#         if ROC_CURVE.exists():
#             st.image(str(ROC_CURVE))
#         else:
#             st.warning("ROC curve not found.")

#     st.divider()

#     st.subheader("Top Feature Importance")

#     if FEATURE_IMPORTANCE.exists():
#         st.image(str(FEATURE_IMPORTANCE))
#     else:
#         st.warning("Feature importance plot not found.")

# # =====================================================
# # EXPLAINABILITY
# # =====================================================

# elif page == "🧠 Explainability":

#     st.title("🧠 Model Explainability")

#     st.markdown("""
# Understanding **why** a machine learning model makes a prediction is essential in
# financial services. SHAP (SHapley Additive exPlanations) helps explain the
# contribution of each feature to the model's predictions, improving transparency
# and supporting responsible lending decisions.
# """)

#     st.divider()

#     if SHAP_SUMMARY.exists():

#         st.subheader("SHAP Summary Plot")

#         st.image(
#             str(SHAP_SUMMARY),
#             use_container_width=True,
#         )

#     else:
#         st.warning("SHAP summary plot not found.")

#     st.divider()

#     st.subheader("Key Interpretation")

#     st.markdown("""
# - Features at the top have the greatest influence on credit risk prediction.
# - Each point represents one customer.
# - Red points indicate higher feature values, while blue points indicate lower feature values.
# - The horizontal position shows whether a feature increases or decreases the predicted credit risk.
# """)

#     st.success(
#         "Model explainability helps build trust by making predictions more transparent and easier to interpret."
#     )
        


# # =====================================================
# # BUSINESS INSIGHTS
# # =====================================================

# elif page == "💼 Business Insights":

#     st.title("💼 Business Insights")

#     st.markdown("""
# This project predicts customer credit risk using transaction behavior instead of
# traditional credit history. The insights generated by the model can support
# better lending decisions while reducing financial risk.
# """)

#     st.divider()

#     st.subheader("Key Findings")

#     st.success(
#         "The Random Forest model achieved excellent predictive performance "
#         f"with a ROC-AUC score of {metrics['roc_auc']:.4f}."
#     )

#     st.info(
#         "Customer transaction behavior provides valuable information for "
#         "identifying potential high-risk borrowers."
#     )

#     st.info(
#         "Feature engineering using customer transaction summaries (RFM-style "
#         "features) significantly improves model performance."
#     )

#     st.divider()

#     st.subheader("Business Value")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("""
# ### Benefits for Bati Bank

# - Improve loan approval decisions
# - Reduce default risk
# - Support financial inclusion
# - Make consistent credit decisions
# """)

#     with col2:
#         st.markdown("""
# ### Practical Applications

# - Loan eligibility assessment
# - Credit risk monitoring
# - Customer segmentation
# - Portfolio risk management
# """)

#     st.divider()

#     st.subheader("Recommendations")

#     st.markdown("""
# - Continue monitoring customer transaction behavior over time.
# - Periodically retrain the model with new transaction data.
# - Integrate the model into the bank's credit assessment workflow.
# - Use explainable AI techniques to support transparent lending decisions.
# """)



import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide",
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown(
    """
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

div[data-testid="metric-container"]{
    background:#f8fafc;
    border:1px solid #e5e7eb;
    border-radius:12px;
    padding:18px;
}

h1{
    color:#1f2937;
}

.sidebar .sidebar-content{
    background:#fafafa;
}

</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------

DATA_PATH = Path("data/processed/final_training_data.csv")
METRICS_PATH = Path("reports/metrics.json")

CONFUSION_MATRIX = Path("reports/confusion_matrix.png")
ROC_CURVE = Path("reports/roc_curve.png")
FEATURE_IMPORTANCE = Path("reports/feature_importance.png")
SHAP_SUMMARY = Path("reports/shap_summary.png")


# ----------------------------------------------------
# LOADERS
# ----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


df = load_data()
metrics = load_metrics()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("💳 Credit Risk Dashboard")

st.sidebar.markdown(
"""
Machine Learning Capstone

**Model:** Random Forest

Transaction-based Credit Scoring
"""
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Model Performance",
        "🧠 Explainability",
        "💼 Business Insights",
    ],
)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

def footer():

    st.divider()

    st.caption(
        "Developed by Aisha Hussein • 10 Academy Capstone • Credit Risk Modeling"
    )


# ====================================================
# HOME
# ====================================================

if page == "🏠 Home":

    st.title("💳 Credit Risk Modeling Dashboard")

    st.write(
        """
Predicting customer credit risk using transaction behavior to support
better lending decisions for Bati Bank.
"""
    )

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "📄 Records",
        f"{len(df):,}"
    )

    col2.metric(
        "📊 Features",
        df.shape[1]
    )

    col3.metric(
        "📈 ROC-AUC",
        f"{metrics['roc_auc']:.4f}"
    )

    col4.metric(
        "🤖 Model",
        "Random Forest"
    )

    st.divider()

    left,right = st.columns(2)

    with left:

        st.subheader("Business Problem")

        st.write(
            """
Many customers lack traditional credit histories, making loan approval
challenging. This project predicts credit risk using transaction
behavior, enabling more inclusive and informed lending decisions.
"""
        )

    with right:

        st.subheader("Project Workflow")

        st.markdown(
"""
1. Data Collection

2. Data Preprocessing

3. Feature Engineering

4. Model Training

5. Hyperparameter Tuning

6. Explainability

7. Dashboard
"""
        )

    footer()

# ====================================================
# DATASET
# ====================================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Features",
        df.shape[1]
    )

    col3.metric(
        "High Risk",
        f"{df['is_high_risk'].sum():,}"
    )

    col4.metric(
        "Low Risk",
        f"{(df['is_high_risk']==0).sum():,}"
    )

    st.divider()

    st.subheader("Target Distribution")

    counts = df["is_high_risk"].value_counts()

    fig,ax = plt.subplots(figsize=(6,4))

    ax.bar(
        ["Low Risk","High Risk"],
        [counts[0],counts[1]]
    )

    ax.set_ylabel("Customers")

    st.pyplot(fig)

    st.divider()

    with st.expander("Preview Processed Dataset",expanded=True):

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

    footer()

# ====================================================
# MODEL
# ====================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("Accuracy",f"{metrics['accuracy']:.4f}")
    c2.metric("Precision",f"{metrics['precision']:.4f}")
    c3.metric("Recall",f"{metrics['recall']:.4f}")
    c4.metric("F1",f"{metrics['f1_score']:.4f}")
    c5.metric("ROC-AUC",f"{metrics['roc_auc']:.4f}")

    st.divider()

    st.subheader("Best Hyperparameters")

    st.table(
        pd.DataFrame(
            {
                "Parameter":[
                    "n_estimators",
                    "max_depth",
                    "min_samples_split"
                ],
                "Value":[
                    200,
                    "None",
                    2
                ]
            }
        )
    )

    st.divider()

    left,right = st.columns(2)

    with left:

        st.subheader("Confusion Matrix")

        if CONFUSION_MATRIX.exists():

            st.image(CONFUSION_MATRIX,use_container_width=True)

    with right:

        st.subheader("ROC Curve")

        if ROC_CURVE.exists():

            st.image(ROC_CURVE,use_container_width=True)

    st.divider()

    st.subheader("Feature Importance")

    if FEATURE_IMPORTANCE.exists():

        st.image(
            FEATURE_IMPORTANCE,
            use_container_width=True
        )

    footer()

# ====================================================
# SHAP
# ====================================================

elif page == "🧠 Explainability":

    st.title("🧠 Model Explainability")

    st.write(
        """
SHAP values explain how each feature contributes to the model's predictions,
making the credit risk model more transparent and interpretable.
"""
    )

    if SHAP_SUMMARY.exists():

        st.image(
            SHAP_SUMMARY,
            use_container_width=True
        )

    st.info(
"""
**How to interpret the plot**

• Features near the top have the strongest influence.

• Red indicates higher feature values.

• Blue indicates lower feature values.

• Horizontal spread shows impact on the prediction.
"""
    )

    footer()

# ====================================================
# BUSINESS
# ====================================================

elif page == "💼 Business Insights":

    st.title("💼 Business Insights")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Key Findings")

        st.success(
            f"Random Forest achieved a ROC-AUC of {metrics['roc_auc']:.4f}."
        )

        st.write(
"""
- Transaction behaviour is a strong predictor of credit risk.
- Feature engineering significantly improved predictive performance.
- The model can support more consistent lending decisions.
"""
        )

    with col2:

        st.subheader("Business Recommendations")

        st.write(
"""
- Support loan approval decisions.

- Reduce default risk.

- Retrain the model periodically.

- Integrate the model into the lending workflow.

- Monitor model performance over time.
"""
        )

    footer()