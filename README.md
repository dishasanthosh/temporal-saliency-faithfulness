# temporal-saliency-faithfulness

# A Faithfulness-Based Evaluation of Temporal Saliency Methods for Time Series Models

> **Evaluating whether AI explanations actually reflect what time-series models rely on.**

This project evaluates the **faithfulness of explainable AI (XAI) methods for time-series models** across clinical and wearable-sensor datasets.

The study compares **SHAP, Gradient Saliency, and WinIT** across two datasets and two neural architectures, using a perturbation-based evaluation framework to measure whether highly ranked timesteps actually influence model predictions.

The central finding is that **Gradient Saliency and WinIT are substantially more faithful than SHAP for well-generalized time-series models**, while faithfulness evaluation itself depends critically on the underlying model's ability to generalize.

---

## Motivation

Explainability methods are increasingly used alongside machine-learning models in high-stakes applications such as healthcare, predictive maintenance, and wearable sensing.

However, an explanation can be **plausible without being faithful**.

For a time-series model, an explanation should identify the timesteps that the model actually relies on. This project therefore asks:

> **If an explanation identifies important timesteps, does removing those timesteps actually degrade the model's performance more than removing supposedly unimportant timesteps?**

This provides an empirical way to evaluate explanation quality rather than assuming that an explanation is trustworthy because it comes from a popular XAI method.

---

## Research Questions

### RQ1

**Which temporal saliency method produces the most faithful explanations for time-series models?**

### RQ2

**Are method rankings stable across datasets and model architectures?**

---

##  Experimental Setup

The benchmark evaluates:

* **2 datasets**
* **2 model architectures**
* **3 explainability methods**
* **12 experimental configurations**

### Datasets

| Dataset            | Domain           | Task                     | Data                                                  |
| ------------------ | ---------------- | ------------------------ | ----------------------------------------------------- |
| **PhysioNet 2012** | Clinical         | ICU mortality prediction | 4,000 patient records, 48-hour sequences, 78 features |
| **PAMAP2**         | Wearable sensors | Activity recognition     | 9 subjects, 12 activities, 52 sensor channels         |

PhysioNet contains 48 hours of clinical measurements including vital signs and laboratory values, while PAMAP2 contains wearable inertial-sensor measurements collected during physical activities.

### Model Architectures

#### LSTM

A 2-layer Long Short-Term Memory network with hidden size 64.

The LSTM models temporal dependencies through recurrent hidden-state memory.

#### TCN

A 4-block Temporal Convolutional Network using:

* Dilation factors: `1, 2, 4, 8`
* Kernel size: `3`
* Causal padding

The TCN captures temporal patterns through localized receptive fields.

---

##  Explainability Methods

### 1. SHAP

Kernel-based Shapley-value estimation.

In this experiment, each timestep is treated as a feature.

A pilot experiment showed that **80 background samples produced degenerate attributions**, so the final experiments used **300 background samples**.

### 2. Gradient Saliency

Computes the gradient of the model output with respect to each input timestep.

Advantages:

* Deterministic
* Computationally efficient
* Directly applicable to neural models

The predicted class was used as the target for multiclass tasks.

### 3. WinIT

**Window Importance over Time (WinIT)** is a temporal perturbation method designed to account for dependencies between neighboring timesteps.

Temporal windows are masked and the resulting prediction change is measured using KL divergence.

Window sizes:

* PhysioNet: ±4 timesteps
* PAMAP2: ±8 timesteps

---

##  Faithfulness Evaluation

The project uses a perturbation-based evaluation framework based on Samek et al. (2017).

For each explanation method:

1. Generate saliency scores for test samples.
2. Average absolute saliency across features and samples.
3. Rank timesteps from most to least important.
4. Iteratively remove the **most important** timesteps.
5. Measure model performance degradation using AUROC.
6. Repeat by removing the **least important** timesteps first.
7. Compare the two degradation curves.

The faithfulness score is:

```text
Faithfulness =
AUC(least-important removed)
-
AUC(most-important removed)
```

A higher score indicates that removing the timesteps identified as important causes greater model degradation, suggesting a more faithful explanation.

Masked values were replaced with the training mean, which becomes zero after standardization. A random saliency baseline was also calculated.

---

##  Results

### Model Performance

| Configuration    |      AUROC | Accuracy / AUPRC |
| ---------------- | ---------: | ---------------: |
| PhysioNet + LSTM |     0.8555 |     78% / 0.4582 |
| PhysioNet + TCN  |     0.8558 |     74% / 0.4810 |
| PAMAP2 + LSTM    |     0.8439 |            40.1% |
| PAMAP2 + TCN     | **0.8840** |        **47.2%** |

The PAMAP2 LSTM showed a major generalization problem: validation AUROC reached **0.9998**, while test accuracy was only **40%**, indicating overfitting to the validation subject.

### Faithfulness Scores

| Configuration    |   Gradient |      WinIT |    SHAP |  Random |
| ---------------- | ---------: | ---------: | ------: | ------: |
| PhysioNet + LSTM | **0.1682** | **0.1676** |  0.0119 |  0.0039 |
| PhysioNet + TCN  | **0.1894** | **0.1866** | -0.0225 |  0.0039 |
| PAMAP2 + TCN     | **0.2458** | **0.2459** |  0.0078 |  0.0038 |
| PAMAP2 + LSTM    |    -0.0019 |    -0.1381 |  0.0122 | -0.0009 |

### Key Result

**Gradient Saliency and WinIT consistently outperformed SHAP on well-generalized models.**

Their faithfulness scores ranged from approximately **0.168 to 0.246**, compared with SHAP scores ranging from approximately **-0.023 to 0.012**.

---

##  Key Findings

### 1. Gradient Saliency and WinIT were the most faithful

Across the well-generalized configurations, Gradient Saliency and WinIT produced very similar and substantially higher faithfulness scores.

Their convergence suggests that two fundamentally different approaches can identify similar critical temporal regions.

### 2. SHAP performed poorly for temporal saliency

SHAP never meaningfully outperformed the random baseline.

The study suggests that a key issue is the mismatch between SHAP's treatment of timesteps as independent features and the sequential dependencies exploited by LSTM and TCN architectures.

### 3. SHAP was highly sensitive to its parameters

On PhysioNet + LSTM:

```text
80 background samples   →   -0.085
300 background samples  →   +0.012
```

Changing only the background sample count moved SHAP from a negative score to slightly above the random baseline.

By contrast, Gradient Saliency and WinIT were deterministic.

### 4. Model generalization is a prerequisite for XAI evaluation

The PAMAP2 + LSTM experiment produced poor faithfulness across **all three methods**.

The model had overfit the validation subject, suggesting that when the underlying model does not learn transferable representations, explanation faithfulness cannot be meaningfully assessed.

This establishes an important practical principle:

> **Validate model generalization before evaluating or trusting explanations.**

### 5. TCN produced higher faithfulness on well-generalized models

TCN achieved higher faithfulness scores than LSTM on both datasets:

```text
PhysioNet:
Gradient — TCN 0.189
Gradient — LSTM 0.168

PAMAP2:
Gradient — TCN 0.246
Gradient — LSTM ≈ 0
```

The study attributes this in part to TCN's localized temporal receptive fields, which can concentrate importance into sharper temporal windows.

---

## 🏥 Clinical Interpretation

On the PhysioNet mortality task, Gradient Saliency and WinIT converged on clinically meaningful features.

The **Glasgow Coma Scale (GCS)** was ranked as the top feature by both methods across LSTM and TCN models.

Other consistently important features included:

* BUN
* Bilirubin
* Age

Both methods concentrated temporal importance toward approximately **hours 35–48** of the 48-hour observation window, suggesting that the models placed greater importance on the recent clinical trajectory.

---

##  Conclusions

The study answers the research questions as follows:

### RQ1

**Gradient Saliency and WinIT are the most faithful methods**, provided that the underlying model generalizes well.

Both substantially outperform SHAP and the random baseline.

### RQ2

Method rankings are **stable when models generalize well**, but become unstable when the underlying model overfits.

Therefore, model-quality verification should precede faithfulness evaluation.

---

##  Contributions

This project contributes:

1. **A controlled benchmark**
   12 experimental configurations spanning two datasets, two architectures, and three explanation methods.

2. **Empirical evaluation of temporal XAI**
   Quantitative comparison of SHAP, Gradient Saliency, and WinIT using faithfulness rather than predictive performance alone.

3. **A model-quality boundary condition**
   Demonstration that poor model generalization can cause all explanation methods to fail.

4. **A reproducible evaluation framework**
   A pipeline covering model training, explanation generation, perturbation curves, and faithfulness scoring.

---

##  Tech Stack

* Python
* PyTorch
* Captum
* SHAP
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* LSTM
* Temporal Convolutional Networks (TCN)

---

##  Project Structure

```text
.
├── data/
│   ├── physionet/
│   └── pamap2/
│
├── models/
│   ├── lstm.py
│   └── tcn.py
│
├── explainers/
│   ├── gradient_saliency.py
│   ├── shap_explainer.py
│   └── winit.py
│
├── evaluation/
│   ├── faithfulness.py
│   ├── perturbation.py
│   └── baselines.py
│
├── experiments/
│   ├── physionet_lstm.py
│   ├── physionet_tcn.py
│   ├── pamap2_lstm.py
│   └── pamap2_tcn.py
│
├── results/
│   ├── faithfulness_scores.csv
│   └── figures/
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

> **Note:** The structure above reflects a recommended organization for the project. Update filenames and directories to match the actual repository implementation.

---

##  Reproducing the Experiments

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare datasets

Download and preprocess:

* PhysioNet 2012
* PAMAP2

Place the processed datasets in the appropriate `data/` directories.

### 5. Train models

```bash
python experiments/physionet_lstm.py
python experiments/physionet_tcn.py
python experiments/pamap2_lstm.py
python experiments/pamap2_tcn.py
```

### 6. Generate explanations

Run the corresponding explanation pipelines for:

```text
SHAP
Gradient Saliency
WinIT
```

### 7. Evaluate faithfulness

```bash
python evaluation/faithfulness.py
```

The evaluation produces degradation curves and faithfulness scores for comparison across methods.

---

##  Evaluation Pipeline

```text
                 ┌──────────────────┐
                 │   Time Series    │
                 │      Dataset     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Train LSTM / TCN │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Verify Model     │
                 │ Generalization   │
                 └────────┬─────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          ┌──────┐   ┌──────────┐   ┌───────┐
          │ SHAP │   │ Gradient │   │ WinIT │
          └──┬───┘   └────┬─────┘   └───┬───┘
             │            │             │
             └────────────┼─────────────┘
                          ▼
                 ┌──────────────────┐
                 │ Rank Timesteps   │
                 │ by Importance     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Perturb / Mask   │
                 │ Timesteps        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ AUROC Degradation│
                 │ Curves           │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Faithfulness     │
                 │ Score             │
                 └──────────────────┘
```

---

##  Important Caveat

The results should not be interpreted as meaning that an explanation method is universally faithful.

Faithfulness depends on the interaction between:

* the explanation method,
* the model architecture,
* the dataset,
* temporal structure,
* hyperparameters, and
* model generalization.

The PAMAP2 LSTM experiment demonstrates this boundary condition particularly clearly.

---

##  References

* Adebayo et al. (2018). *Sanity Checks for Saliency Maps.*
* Doshi-Velez & Kim (2017). *Towards a Rigorous Science of Interpretable Machine Learning.*
* Ismail et al. (2020). *Benchmarking Deep Learning Interpretability in Time Series Predictions.*
* Kokhlikyan et al. (2020). *Captum: A Unified and Generic Model Interpretability Library for PyTorch.*
* Leung et al. (2023). *Temporal Dependencies in Feature Importance for Time Series Prediction.*
* Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.*
* Samek et al. (2017). *Evaluating the Visualization of What a Deep Neural Network Has Learned.*
* Silva et al. (2012). *Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012.*

See the project report for the complete reference list.

---

##  Author

**Disha Santhosh**

Practicum in Information Management
Information School, University of Washington
June 2026

---

##  Key Takeaway

> **For time-series XAI, an explanation should not be trusted simply because it looks reasonable. It should be tested against the model's behavior.**

This project demonstrates that **Gradient Saliency and WinIT can provide substantially more faithful temporal explanations than SHAP**, while also showing that **explanation quality is inseparable from model generalization quality**.
