# EstateIQ — Machine Learning Candidate Evaluation Report (Phase 4B)

**Date of Execution:** 2026-10-21  
**Dataset Used:** `ml/training/data/processed/train_cleaned.csv`  
**Dataset SHA-256:** `f667746b9afd1a85c7b1f96c7ca501c8d6174893572828f57049614ad6d27cc0`  
**Split Configuration:** 80% Training (23,240 rows), 20% Validation (5,810 rows), `random_state=42`  
**Preprocessed Features:** 59 columns (fitted exclusively on training split)  

---

## 1. Candidate Model Benchmark Results

All eight configurations were trained on the training partition and evaluated on the exact same validation partition.
Metrics are strictly computed in the original price-in-lakhs scale (1 Lac = 100,000 INR).

| Rank | Model Configuration | Target Mode | Val RMSE (Lacs) | Val MAE (Lacs) | Val R² Score | Train RMSE (Lacs) | Train MAE (Lacs) | Train R² | Overfitting Ratio (Val/Train RMSE) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **DecisionTreeRegressor (max_depth=12)** | **Raw** | **253.16** | **40.70** | **0.8830** | **158.19** | **30.78** | **0.9360** | **1.60** |
| 2 | RandomForestRegressor (100 trees, depth=15) | Raw | 326.46 | 40.43 | 0.8054 | 192.75 | 27.27 | 0.9050 | 1.69 |
| 3 | KNeighborsRegressor (k=7) | Raw | 391.82 | 50.01 | 0.7196 | 247.47 | 37.05 | 0.8436 | 1.58 |
| 4 | DecisionTreeRegressor (max_depth=12) | log1p | 413.59 | 41.03 | 0.6876 | 256.49 | 28.53 | 0.8319 | 1.61 |
| 5 | RandomForestRegressor (100 trees, depth=15) | log1p | 423.03 | 37.45 | 0.6732 | 265.17 | 25.43 | 0.8203 | 1.60 |
| 6 | KNeighborsRegressor (k=7) | log1p | 467.22 | 49.54 | 0.6013 | 303.22 | 34.61 | 0.7661 | 1.54 |
| 7 | LinearRegression (OLS) | Raw | 601.59 | 130.01 | 0.3391 | 555.41 | 129.80 | 0.2098 | 1.08 |
| 8 | LinearRegression (OLS) | log1p | 8,012.89 | 206.33 | -116.2576 | 8,741.32 | 199.12 | -99.4710 | 0.92 |

---

## 2. Model Selection Rationale

### Primary Selection Metric: Validation RMSE
* In accordance with `SRS.md` Section 5.3, the final model is selected by **minimizing validation RMSE on the original price scale**.
* **`DecisionTree_Raw` achieved the lowest validation RMSE (253.16 Lacs)** and the highest $R^2$ score (**0.8830**), outperforming all linear, ensemble, and distance-based competitors on squared-error loss.
* Validation MAE for `DecisionTree_Raw` is **40.70 Lacs**, within $0.27$ Lacs of Random Forest ($40.43$ Lacs).

### Overfitting & Generalization Analysis
* `DecisionTree_Raw` train RMSE is $158.19$ Lacs vs. validation RMSE of $253.16$ Lacs (overfitting ratio $1.60$).
* Setting `max_depth=12`, `min_samples_split=10`, and `min_samples_leaf=5` effectively bounded tree growth and prevented pure memorization of training instances while capturing the non-linear interactions between area, locality/city, and price.

---

## 3. Analysis of Target Transformation (`log1p` vs. Raw)

1. **Why `log1p` caused higher RMSE in Original Scale:**
   * While `RandomForest_Log1p` attained the lowest validation MAE ($37.45$ Lacs vs. $40.43$ Lacs), its validation RMSE was significantly worse ($423.03$ Lacs vs. $326.46$ Lacs).
   * **Root Cause:** When predictions are modeled in logarithmic space, small absolute errors in high-value luxury tiers (e.g., predicting $\log(y) = 8.5$ instead of $8.0$) are magnified exponentially upon $\text{expm1}$ back-transformation ($\Delta \approx 1,940$ Lacs), penalizing squared error severely.
2. **Catastrophic Failure of Linear Regression with `log1p`:**
   * Linear Regression with `log1p` target yielded negative $R^2$ ($-116.26$) and an extreme RMSE of $8,012.89$ Lacs.
   * Linear regression hyperplanes extrapolate unbounded logits on combinations of large square footage and binary indicators, producing extreme exponents when inverted.
3. **Conclusion:**
   * For the primary project objective (minimizing RMSE on actual property transaction scale), **raw-target training provides superior stability and lower overall squared error**.

---

## 4. Production Candidate Artifact

The winning configuration (`DecisionTree_Raw`) and the fitted feature preprocessor were bound into a single self-contained inference container:

* **Artifact Path:** `ml/inference/artifacts/estateiq_model_pipeline.joblib` (85.7 KB)
* **Metadata Path:** `ml/inference/artifacts/model_metadata.json`
* **Experiment Report:** `ml/training/experiments/candidate_comparison_report.json`
* **Reload Verification:** Verified by deserializing and executing predictions on validation slices.

---

## 5. Mandatory Limitation Disclaimer

As required by `PRD.md` Section 1 and `agents.md`:
Validation metrics represent empirical performance on the held-out partition of the supplied dataset. They are **not guarantees of real-world real estate transaction prices or official market valuations**.
