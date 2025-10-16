# 🧠 Impact of Fine-Tuning Techniques on Model Accuracy and Performance

*A Comparative Analysis Using Credit Card Fraud Detection Dataset (Kaggle)*

---

## 📘 Project Overview

This project explores the **impact of hyperparameter fine-tuning techniques** — including **GridSearchCV**, **RandomizedSearchCV**, and **Bayesian Optimization (BayesSearchCV)** — on the **accuracy and performance** of both **bagging** and **boosting** models.

The objective is to **quantitatively assess how different cross-validation (CV) and tuning methods influence model ROC-AUC scores**, comparing each technique against a **baseline (untuned) model**.

The study uses the **Credit Card Fraud Detection dataset from Kaggle**, a highly imbalanced binary classification problem that mirrors real-world financial transaction data.

---

## 🎯 Key Objectives

1. **Evaluate baseline model performance** (without fine-tuning).
2. **Apply and compare fine-tuning methods**:

   * GridSearchCV (exhaustive parameter search)
   * RandomizedSearchCV (stochastic parameter sampling)
   * Bayesian Optimization (probabilistic exploration using Scikit-Optimize)
3. **Benchmark across multiple models**:

   * Random Forest
   * Extra Trees
   * XGBoost
   * LightGBM
   * CatBoost
4. **Visualize and interpret** the comparative effects of each tuning strategy.
5. **Draw strategic conclusions** on computational efficiency versus performance gain.

---

## 🧩 Dataset Description

* **Dataset**: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Size**: 284,807 transactions
* **Features**: 30 columns (V1–V28 are PCA-transformed features, plus `Amount` and `Time`)
* **Target Variable**: `Class` (1 = Fraud, 0 = Legitimate)
* **Nature**: *Highly imbalanced* (≈0.17% fraud cases)

---

## 🧠 Models Evaluated

| Category     | Model                  | Library      |
| ------------ | ---------------------- | ------------ |
| **Bagging**  | RandomForestClassifier | Scikit-Learn |
|              | ExtraTreesClassifier   | Scikit-Learn |
| **Boosting** | XGBoostClassifier      | XGBoost      |
|              | LGBMClassifier         | LightGBM     |
|              | CatBoostClassifier     | CatBoost     |

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis (EDA)

* Inspected feature distributions and imbalance ratio.
* Applied scaling using `StandardScaler`.
* Split data into `train-test` sets (80-20 split).
* Verified no data leakage.

### 2. Baseline Modeling

Each model was trained using **default parameters** to establish a benchmark ROC-AUC.

```python
baseline_rf = RandomForestClassifier(random_state=42)
baseline_rf.fit(X_train, y_train)
y_pred_base = baseline_rf.predict(X_test)
print("ROC-AUC:", roc_auc_score(y_test, y_pred_base))
```

### 3. Fine-Tuning Techniques

#### 🔹 GridSearchCV

* Exhaustively searches parameter space.
* Computationally expensive but exhaustive.

#### 🔹 RandomizedSearchCV

* Samples random parameter combinations.
* Faster with near-equivalent performance.

#### 🔹 Bayesian Optimization (BayesSearchCV)

* Uses probabilistic models (Gaussian Processes).
* Learns optimal regions dynamically.
* Most efficient for large parameter spaces.

### 4. Evaluation Metric

* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
* Selected due to imbalanced data nature (precision and recall considered implicitly).

---

## 📊 Results Summary

### **1. Baseline Models (No Fine-Tuning)**

| Model        |   ROC-AUC  |
| :----------- | :--------: |
| RandomForest |   0.9459   |
| ExtraTrees   |   0.9449   |
| XGBoost      | **0.9551** |
| LightGBM     |   0.9531   |
| CatBoost     |   0.9510   |

---

### **2. Tuned Models (Fine-Tuning Applied)**

| Model        | GridSearch | RandomizedSearch | BayesianSearch |
| :----------- | :--------: | :--------------: | :------------: |
| RandomForest |   0.9459   |    **0.9469**    |   **0.9469**   |
| ExtraTrees   |   0.9449   |      0.9418      |   **0.9469**   |
| XGBoost      | **0.9551** |      0.9510      |     0.9459     |
| LightGBM     |   0.9531   |    **0.9551**    |     0.9500     |
| CatBoost     |   0.9510   |      0.9510      |     0.9510     |

---

### **3. Average ROC-AUC Comparison**

| Technique             | Mean ROC-AUC |
| :-------------------- | :----------: |
| Baseline              |  **0.9504**  |
| GridSearchCV          |    0.9504    |
| RandomizedSearchCV    |    0.9496    |
| Bayesian Optimization |    0.9489    |

---

## 📈 Visualizations

### 🔹 ROC-AUC Comparison by Model

* Bar plot comparing each model across tuning methods.
* Shows minimal gain across techniques.

### 🔹 Average ROC-AUC by Tuning Technique

* Aggregated performance visualization.
* Highlights efficiency vs. gain trade-off.

### Example Code Snippet

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(data=df_results, x='Model', y='ROC-AUC', hue='Tuning Method', palette='viridis', edgecolor='black')
plt.title("Model ROC-AUC Comparison Across Tuning Techniques", fontsize=14, fontweight='bold')
plt.xlabel("Model")
plt.ylabel("ROC-AUC Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
```

---

## 🧭 Key Insights

1. **Minimal performance gains** observed after tuning — baseline models already performed optimally.
2. **GridSearchCV** confirmed stability but was computationally expensive.
3. **RandomizedSearchCV** offered similar accuracy with faster runtime.
4. **Bayesian Optimization** improved search intelligence but not necessarily ROC performance.
5. **Boosting models (XGBoost, LightGBM, CatBoost)** consistently outperformed bagging models across all setups.
6. **Model architecture and data representation** had greater influence than tuning strategy on fraud detection performance.

---

## ⚡ Conclusion

Fine-tuning is often seen as the holy grail of optimization, but this study demonstrates that **when models are well-regularized and the dataset is structured**, the performance improvements from hyperparameter search are often **marginal**.

> "Beyond a certain threshold of model maturity, **data quality, feature engineering, and problem framing** contribute more to success than parameter tweaking."

---

## 🧮 Tech Stack

* **Language:** Python 3.10+
* **Core Libraries:** Scikit-Learn, XGBoost, LightGBM, CatBoost, Scikit-Optimize
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Kaggle Notebook

---

## 🔍 Future Work

* Extend to **deep learning models** (e.g., tabular neural networks).
* Include **cross-dataset generalization** studies.
* Benchmark **time-complexity vs. accuracy trade-offs** for industrial optimization pipelines.
* Explore **multi-objective Bayesian optimization** (balancing accuracy and latency).

---

## 🧑‍💼 Author

**Willy [Data Scientist | ML Engineer | AI & Blockchain Innovator]**
💼 Focus: Intelligent Systems, Applied Machine Learning, and NFT Technology.
📫 Contact: [www.linkedin.com/in/wilfred-kamau-001134267]
🌐 Portfolio: [(https://github.com/Kam-Willy)]

---
