# EstateIQ — Data Preprocessing Decisions & Architecture

**Document Version:** 1.0  
**Status:** Approved for Implementation (Phase 4A)  
**Applies to:** ML Training, Inference Services, Data Pipeline  

---

## 1. Target Derivation & Leakage Prevention

* **Target Column:** `TARGET(PRICE_IN_LACS)` is derived exclusively from `train_cleaned.csv`.
* **Explicit Separation:** `separate_features_and_target_records()` and `load_and_split_training_data()` extract `y` immediately upon reading the data. `X` is strictly stripped of `TARGET(PRICE_IN_LACS)`.
* **Target Leakage Guard:** `TargetLeakageGuard` is placed as the first step of the feature pipeline. If the target column is detected anywhere in the input schema or feature matrix, it immediately raises a hard `ValueError`.
* **Prohibition on Evaluation with `test.csv`:** Both `test.csv` and `sample_submission.csv` contain no ground-truth targets. The loader explicitly raises an exception if any attempt is made to use test or submission files for supervised training or validation.

---

## 2. Address Inspection & City Feature Extraction

### 2.1 Address Inspection Findings
* Every record in both `train_cleaned.csv` (29,050 rows) and `test_cleaned.csv` (68,720 rows) contains at least one comma (0 single-token addresses).
* Multi-comma addresses (e.g., `Suddaguntepalya, C V Raman Nagar, Bangalore`) consistently place the municipal city name at the final comma token.
* Exactly 256 unique cities were extracted from the training set.
* In the test set, 293 unique cities exist, of which **59 cities never appeared in the training set**.

### 2.2 Extraction & Encoding Rule
* **Extraction Function:** `extract_city_from_address(address)` splits by comma and takes the final non-empty token, stripped of leading/trailing whitespace.
* **Frequency Thresholding (`CityExtractor`):** Cities appearing with frequency $\ge 50$ (default `CITY_MIN_FREQUENCY`) during `fit()` on the training partition are retained. The top ~40 cities account for over 90.5% of the total dataset.
* **Unseen City Safety:** Any city with $< 50$ occurrences or unseen during inference/validation is mapped to `"Other"`.
* **One-Hot Encoding:** Downstream `OneHotEncoder` uses `handle_unknown='ignore'`, preventing runtime errors when encountering new categorical values.

---

## 3. Multicollinearity Resolution: `UNDER_CONSTRUCTION` vs. `READY_TO_MOVE`

* **Audit Evidence:** Across all 29,050 records, $\text{UNDER\_CONSTRUCTION} + \text{READY\_TO\_MOVE} = 1$ in 100% of rows (0 mismatches).
* **Mathematical Concern:** The two features are perfect linear inverses ($RTM = 1 - UC$). Retaining both creates a singular covariance matrix ($X^T X$ is rank-deficient), causing infinite variance in Ordinary Least Squares (OLS) regression coefficients.
* **Decision:** Retain `UNDER_CONSTRUCTION` as the canonical binary indicator and drop `READY_TO_MOVE` via `CollinearFeatureDropper`.

---

## 4. Geographic Coordinate Validation & Missing Imputation

* **Coordinate Reversal in Raw Data:** Audited data revealed `LONGITUDE` and `LATITUDE` headers were reversed in raw files. Processed data corrected them to their true positions.
* **Bounding Box Validation:** Coordinates are validated against India's geographic bounds:
  * Latitude: $6.5^\circ \text{N}$ to $37.5^\circ \text{N}$
  * Longitude: $68.0^\circ \text{E}$ to $97.5^\circ \text{E}$
* **Missing Value Handling:** Coordinates outside these bounds or blank (255 rows in cleaned train, 540 in cleaned test) are set to `np.nan`.
* **Imputation Strategy:** A `SimpleImputer(strategy='median')` fits median coordinates strictly on `X_train`. Missing validation or inference coordinates receive the training median, preserving geographic validity without synthetic data generation.

---

## 5. Extreme Area & Price Values Investigation

* **Observed Extremes:**
  * Minimum `SQUARE_FT`: $3.0$ sq ft (26 properties $< 100$ sq ft).
  * Maximum `SQUARE_FT`: $254,545,454.5$ sq ft (68 properties $> 100,000$ sq ft).
  * Maximum Price: $30,000.0$ Lacs (₹300 Crore; 289 properties $> 1,000$ Lacs).
* **Policy Compliance:** In accordance with `PRD.md` and `agents.md`, no records are deleted arbitrarily solely for being extreme, and no synthetic values are invented.
* **Scaling Decision:** Standard min-max normalization ($[0, 1]$) would collapse over 99% of valid properties into a near-zero range $[0, 0.0001]$ due to the 254M outlier. Therefore, `RobustScaler` (based on interquartile range $IQR = Q_3 - Q_1$ and median) is used for numeric features.

---

## 6. Target Transformation Analysis: Log-Scale Benefits vs. Trade-Offs

Per project instructions, we do not prematurely commit to a log-target transformation. Both options will be evaluated empirically during candidate model benchmarking (Phase 4B).

### Benefits of $\log(1 + y)$ Transformation:
1. **Variance Stabilization:** Compresses the long right tail ($30,000$ Lacs max vs. $61.8$ Lacs median), preventing extreme luxury properties from dominating squared-error gradients ($MSE \propto (y - \hat{y})^2$).
2. **Relative Error Alignment:** Optimizing RMSE on log-price corresponds approximately to minimizing Mean Absolute Percentage Error (MAPE), which aligns closely with how real-estate markets evaluate price errors (percentage discrepancy).
3. **Non-Negative Guarantee:** Taking the inverse $\exp(\hat{y}_{log}) - 1$ guarantees price predictions remain strictly positive ($\hat{y} \ge 0$).

### Trade-Offs and Drawbacks:
1. **Metric Distortion:** A model optimized for log-RMSE minimizes geometric error rather than linear rupee loss. When converted back to Lacs, the model may underpredict ultra-high-end properties.
2. **Jensen's Inequality Bias:** For any random variable $X$, $\mathbb{E}[\exp(X)] \ge \exp(\mathbb{E}[X])$. Directly exponentiating log predictions can introduce a downward bias in the expected mean price unless corrected (smearing estimate).
3. **Evaluation Standard:** Final benchmark metrics (MAE, RMSE) must be computed and compared on the **original, unscaled price scale (Lacs)** for all models to ensure fair comparison.

---

## 7. Reusability for Inference

The entire feature pipeline (`ml.pipeline.preprocessing.build_feature_preprocessor()`) accepts both raw pandas DataFrames and raw dictionary payloads (as validated by `PropertyInputSchema` in FastAPI).
When fitted on `X_train`, the complete pipeline (including imputer medians, scaler parameters, city lists, and one-hot encoder categories) is bundled into a single Joblib artifact during training, ensuring exact parity between training and inference.
