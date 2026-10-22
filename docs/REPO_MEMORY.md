# EstateIQ — Repository Memory & Engineering System State

**System:** EstateIQ — An Intelligent Real Estate Price Prediction System Using Machine Learning  
**Document ID:** EIQ-MEM-001  
**Last Updated:** 2026-10-22  
**Status:** Phases 1 to 5.3 Complete & Verified (38/38 Tests); Phase 6 (Frontend) Ready for Execution  
**Lead Developer:** srush  
**Assistant:** Antigravity AI  

---

## 1. Absolute Security Directive — Zero `.env` Access

> [!CAUTION]
> **MANDATORY SECURITY DIRECTIVE: ZERO `.env` ACCESS**  
> Under no circumstances may any agent or tool open, read, parse, search, display, copy, edit, rename, move, delete, or inspect any actual `.env` file (`.env`, `.env.local`, `.env.production`, or any file containing active secrets).
>
> 1. Never execute shell commands, scripts, ripgrep queries, or IDE tools targeting `.env` files.
> 2. Never inspect live environment variable values via process memory, debuggers, logs, or command-line prints.
> 3. Never disclose, summarize, copy, or print secrets in generated code, documentation, reports, or logs.
> 4. For configuration definitions and environment schemas, rely **exclusively** on safe documentation and template files with dummy placeholders (e.g., [`.env.example`](file:///d:/EstateIQ/.env.example)).
> 5. If any task appears to require reading real `.env` secrets, **stop immediately** and request human instructions from srush. Never bypass this restriction.

---

## 2. Product Mission & System Scope

EstateIQ is a full-stack, configuration-driven educational and decision-support web application that provides:
1. **Accurate Indian Real Estate Valuation:** Estimates property prices in Lakhs (₹1,00,000) and Crores (₹1,00,00,000) using property attributes (BHK, Square Footage, Construction Status, RERA approval, Resale status, Coordinates, City).
2. **Model Transparency & Comparison:** Exposes genuine validation performance metrics (RMSE, MAE, $R^2$) across 4 regression algorithms (Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbors) evaluated on raw vs log1p targets.
3. **Residual Distribution Insights:** Informs users that 87.01% of validation error is concentrated in the top 1% luxury properties ($> ₹10\text{ Cr}$), while standard market properties ($\le ₹10\text{ Cr}$, 99% of data) exhibit an RMSE of **91.70 Lacs** (with Budget tier MAE of **14.25 Lacs**).
4. **Mandatory Educational Disclaimer:** Explicitly communicates that valuations are statistical estimations, not certified appraisals or financial guarantees.
5. **Private Multi-User Accounts:** Secure registration, authentication, stateful sessions, and tenant-isolated historical property valuation logs.
6. **Dataset Governance:** Real estate dataset CSVs are preserved in local directories (`ml/training/data/`) but excluded from version control pending redistribution licensing verification. Users wishing to run model training must supply the authorized dataset locally.

---

## 3. Approved Technology Stack

| Layer | Technology | Version | Rationale & Configuration |
|---|---|---|---|
| **Backend Framework** | FastAPI | 0.141.1 | Async high-performance REST API with automated OpenAPI docs. |
| **Data Validation** | Pydantic | 2.13.5 | Strict input validation matching real-world property constraints. |
| **ORM & Database** | SQLAlchemy 2.0 | 2.0.54 | Explicit mapped columns, connection pooling, isolated tenant queries. |
| **Database Engine** | PostgreSQL (Supabase) / SQLite | Local SQLite / Supabase PG | Managed via versioned Alembic migrations. |
| **ML Engine** | scikit-learn & joblib | 1.9.1 / 1.6.0 | Preprocessing Pipeline + DecisionTreeRegressor (max_depth=12). |
| **Data Processing** | pandas & numpy | 3.0.6 / 2.5.3 | Data extraction, frequency thresholding, bounding-box checks. |
| **Authentication** | Argon2id & Bcrypt | 25.1.0 / 4.0.1 | Argon2id required for new passwords; Bcrypt backward-compatibility with auto-rehash. |
| **Frontend Framework** | Next.js 14 App Router | Next.js 14, React 18, TS | Tailwind CSS, Lucide React, Recharts visualization. |
| **Testing** | pytest | 9.1.1 | 33/33 passing unit, pipeline, security, and API integration tests. |

---

## 4. Key Architectural & Data Decisions

1. **Collinear Dropper:** Identified that $\text{UNDER\_CONSTRUCTION} + \text{READY\_TO\_MOVE} = 1.0$ across 100% of rows. The pipeline explicitly drops `READY_TO_MOVE`.
2. **City Extraction:** Trailing comma token of `ADDRESS` is extracted. Cities with frequency $\ge 50$ are preserved; all others and unseen cities are mapped to `'Other'`.
3. **Coordinate Validation:** India bounding box ($6.5^\circ \le \text{Lat} \le 37.5^\circ$, $68.0^\circ \le \text{Lon} \le 97.5^\circ$). Out-of-bounds or zero coordinates become `NaN` and are imputed with training medians.
4. **Log1p vs Raw Scale:** Evaluation proved log1p models produce catastrophic rupee errors on multi-crore luxury homes upon $\text{expm1}$ inversion. DecisionTree with raw target achieved superior RMSE (253.16 Lacs) and $R^2$ (0.8830).
5. **No Anonymous Lambdas in Transformers:** In [`ml/pipeline/preprocessing.py`](file:///d:/EstateIQ/ml/pipeline/preprocessing.py), top-level named function `to_float_array` prevents `PicklingError` during serialization.
6. **Stateful Session Store with Hashing at Rest:** Plain session tokens are sent via HttpOnly cookies, but stored as `SHA-256(token)` in `user_sessions`. Database breaches cannot compromise active sessions.
7. **Strict Multi-Tenancy:** Prediction history queries enforce `PredictionHistory.user_id == current_user.id`.

---

## 5. Artifact & Codebase Map

* Model Artifact: [`ml/inference/artifacts/estateiq_model_pipeline.joblib`](file:///d:/EstateIQ/ml/inference/artifacts/estateiq_model_pipeline.joblib)
* Model Metadata: [`ml/inference/artifacts/model_metadata.json`](file:///d:/EstateIQ/ml/inference/artifacts/model_metadata.json)
* Preprocessing Code: [`ml/pipeline/preprocessing.py`](file:///d:/EstateIQ/ml/pipeline/preprocessing.py)
* Training Script: [`ml/training/scripts/train_and_evaluate.py`](file:///d:/EstateIQ/ml/training/scripts/train_and_evaluate.py)
* Backend Entrypoint: [`backend/app/main.py`](file:///d:/EstateIQ/backend/app/main.py)
* Backend Config: [`backend/app/core/config.py`](file:///d:/EstateIQ/backend/app/core/config.py)
* Backend Security: [`backend/app/core/security.py`](file:///d:/EstateIQ/backend/app/core/security.py)
* Backend Services: [`backend/app/services/`](file:///d:/EstateIQ/backend/app/services/)
* Backend Routers: [`backend/app/api/`](file:///d:/EstateIQ/backend/app/api/)
* **Backend Database:** Local PostgreSQL 18.1 (`estateiq_db` on port 5432) managed with Alembic.
* **Test Suite:** [`backend/tests/test_postgres_integration.py`](file:///d:/EstateIQ/backend/tests/test_postgres_integration.py) verifies live PostgreSQL tables, keys, Argon2id auth, sessions, isolation, and prediction persistence.
* **Technical Spec:** [`technical.md`](file:///d:/EstateIQ/technical.md)
* **Progress Tracker:** [`progress.md`](file:///d:/EstateIQ/progress.md)
* **ML Evaluation Report:** [`docs/ML_EVALUATION_REPORT.md`](file:///d:/EstateIQ/docs/ML_EVALUATION_REPORT.md)
* **Preprocessing Decisions:** [`docs/PREPROCESSING_DECISIONS.md`](file:///d:/EstateIQ/docs/PREPROCESSING_DECISIONS.md)

---

## 6. Hand-Off & Next Session Protocol

* **Target User:** Address as **srush**.
* **Current State:** 38/38 pytest tests passing (15 SQLite unit tests, 5 live PostgreSQL integration tests, 12 preprocessing tests, 6 training pipeline tests). Local PostgreSQL `estateiq_db` verified, tables initialized and migrated, clean test teardown completed.
* **Next Action:** Request srush's approval to begin **Phase 6: Next.js Frontend Implementation**.
