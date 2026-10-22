# EstateIQ — Product Requirements Document (PRD)

**Document version:** 1.0  
**Status:** Initial baseline for review  
**Project name:** EstateIQ  
**Formal title:** An Intelligent Real Estate Price Prediction System Using Machine Learning  
**Prepared for:** Project implementation in Google Antigravity

## 1. Product overview

EstateIQ is a full-stack web application that estimates a property's price from user-provided property characteristics using a machine-learning regression model trained and evaluated on the actual dataset supplied for this project. The application will provide a clear prediction experience, transparent model-evaluation information, and a history of a signed-in user's own predictions.

EstateIQ is an educational and decision-support tool. Unless the dataset's provenance establishes otherwise, predictions must be described as estimates based on the dataset's target variable—not as verified market values, valuations, or guaranteed transaction prices.

## 2. Problem statement

Real-estate datasets can contain many property attributes and complex relationships with price. EstateIQ aims to demonstrate an end-to-end ML workflow: data inspection, preprocessing, comparison of regression algorithms, evaluation using genuine validation results, model serving through an API, and a user-friendly interface.

## 3. Goals

1. Build a maintainable full-stack application with a clear separation between frontend, API/backend, ML training, and inference.
2. Train and compare suitable regression models on the user-provided dataset.
3. Select the model based on genuine validation metrics rather than a preselected winner.
4. Let users enter supported property attributes and receive a price estimate from the selected trained model.
5. Explain model performance using real metrics and clear limitations.
6. Store user accounts and prediction history securely in PostgreSQL.
7. Provide repeatable setup, testing, and deployment documentation.

## 4. Non-goals and scope boundaries

- Do not claim that an estimate is an official valuation, financial advice, or a guaranteed sale price.
- Do not generate fake listings, synthetic training records, fake users, fake dashboard totals, or fabricated performance scores.
- Do not build a live property marketplace, payments, broker messaging, map-based discovery, or external listing integrations unless separately approved and supported by genuine data.
- Do not use the unlabeled test CSV to calculate supervised evaluation metrics.
- Do not expose database credentials, model-training internals, or secrets to the browser.
- Do not deploy, push to a remote repository, or mutate a remote database without explicit user approval.

## 5. Target users and roles

### 5.1 Guest
- View public landing page and general explanation of EstateIQ.
- Register or sign in.
- No access to private prediction history.

### 5.2 Registered user
- Sign in and sign out securely.
- Submit property details that match the trained model's input schema.
- View the predicted price, target units, and relevant limitations.
- View only their own prediction history and details.
- Manage basic profile/session functions that are implemented and tested.

### 5.3 Administrator (only if implemented)
- Access explicitly authorized operational or model metadata views.
- Must not automatically gain access to passwords or secrets.
- Any admin dashboard statistics must be derived from actual database records.
- Admin role assignment must be restricted; users must not be able to self-promote.

## 6. Core user journeys

1. A visitor opens EstateIQ and understands that predictions are estimates.
2. The visitor registers and signs in.
3. The user opens the prediction form.
4. The UI displays only inputs supported by the trained model's declared schema.
5. The user submits valid values; the backend validates them and calls the loaded model pipeline.
6. The result page displays the estimate and units/scale consistent with the target variable.
7. If persistence is enabled and the user is authenticated, the prediction is saved and appears in that user's history.
8. The user can inspect model comparison results and see the actual validation metrics produced by the training run.

## 7. Functional requirements

### FR-01: Landing and product information
Provide a responsive landing page describing EstateIQ, its supported workflow, and limitations. Avoid unsupported claims such as guaranteed accuracy.

### FR-02: Authentication
Provide registration, login, logout, session expiry, and appropriate validation/error states. Use the security requirements in the SRS. Password reset may be added only with a properly configured delivery mechanism; do not pretend an email was sent if no provider is configured.

### FR-03: Property input form
Build a clear, accessible form from the final feature schema. Include units, field descriptions, required/optional status, valid ranges based on inspected data/domain rules, and useful validation errors. Do not invent fields or silently substitute missing values.

### FR-04: Prediction
Submit the form to the backend prediction endpoint. Show the returned estimate with the target unit and a note that the result is model-generated and not guaranteed. Handle unavailable model, invalid inputs, and server errors gracefully.

### FR-05: Model training and comparison
Train suitable candidate models using the actual supplied labeled training data. Candidate models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor

Other algorithms may be added only with documented justification. Compare them using a consistent validation strategy and the same eligible data/features. Select the final model using a documented metric-based rule agreed in the SRS/ML report. Report MAE, RMSE, and R² where appropriate. Never hardcode scores.

### FR-06: Model metadata
Display model name/version, training timestamp, dataset fingerprint or provenance metadata where practical, feature schema version, and actual evaluation metrics only when a successful training run produced them. If metrics are unavailable, show a clear unavailable state rather than placeholders.

### FR-07: Prediction history
Authenticated users can view and inspect their own saved predictions. Records should include input snapshot, estimate, model version, timestamp, and relevant schema/version metadata. Enforce ownership in the backend for every request.

### FR-08: Dashboard
Provide a useful dashboard containing real, query-backed information such as the user's prediction count and recent history. Do not seed fake activity or display static sample statistics as if they were real.

### FR-09: Error and empty states
Provide clear states for no history, unavailable model, invalid inputs, API/database failure, and missing training artifacts. Avoid fabricated fallback predictions.

### FR-10: Documentation and reproducibility
Document dataset placement, environment setup, preprocessing decisions, training/evaluation commands, API use, migrations, tests, and deployment. Include no secrets in documentation.

## 8. ML and data principles

- The supplied dataset is the sole training-data source unless the user explicitly approves another real source.
- Keep original source files unchanged; put derived files in a separate processed directory.
- Inspect source/provenance, columns, types, missingness, duplicates, target distribution, suspicious coordinates, outliers, and possible leakage before training.
- Do not treat a CSV as verified transaction data unless its provenance confirms that claim.
- Do not remove extreme records solely because they are extreme; investigate and document decisions.
- Correct data issues only when justified and reproducible. Do not invent replacement values.
- Fit preprocessing only on training partitions and apply the fitted pipeline to validation/inference data.
- Use a fixed random seed where applicable and record it.
- The supplied `test.csv` has no target column, so it cannot produce supervised evaluation scores. Keep it separate for its intended purpose.
- The target column in the currently supplied dataset is named `TARGET(PRICE_IN_LACS)`. Confirm its meaning, scale, and units from the source documentation before displaying currency labels.
- Avoid presenting the model's regression metrics as “accuracy percentage.”

## 9. UI/UX requirements

- Professional, responsive layout for desktop, tablet, and mobile.
- Consistent typography, spacing, components, loading states, and form validation.
- Accessible labels, keyboard navigation, visible focus, semantic structure, and adequate contrast.
- Useful visualizations only when backed by actual evaluation/history data.
- Do not copy a complete third-party design or use unlicensed assets.
- Preserve existing approved behavior and do not add unrelated features.

## 10. Technology baseline

- Frontend: Next.js, TypeScript, Tailwind CSS.
- Charts: Recharts where useful and data exists.
- Backend: Python, FastAPI, Pydantic, Uvicorn.
- ML: Pandas, NumPy, Scikit-learn, Matplotlib where useful, Joblib.
- Database: Supabase-hosted PostgreSQL.
- ORM: SQLAlchemy 2.x.
- Migrations: Alembic.
- Authentication: custom FastAPI authentication, secure password hashing (Argon2), secure HttpOnly cookies, CSRF protection, session expiry, rate limiting, and role-based authorization.
- Tests: pytest; frontend linting/type checking and relevant component/API tests.
- Deployment target: Vercel frontend, Render backend, Supabase PostgreSQL, subject to configuration and user approval.

## 11. Acceptance criteria

The product is ready for project acceptance when:
1. The application starts from documented instructions without relying on fabricated data.
2. The dataset inspection report identifies schema, quality issues, and documented decisions.
3. Training runs successfully on the supplied labeled data, or clearly reports blockers without claiming success.
4. Candidate models are evaluated consistently and actual metrics are saved.
5. The chosen model and preprocessing are packaged together and loaded by the API.
6. The prediction API rejects invalid inputs and returns predictions only when a valid trained artifact is available.
7. Frontend forms and results are connected to the actual API.
8. Authentication and prediction-history ownership checks pass tests.
9. Migrations can be applied to a local/test database and are not silently applied to production.
10. Automated tests, linting, and type checks are run and their actual outcomes documented.
11. There are no fake dashboard values, mock property listings, fabricated model scores, or exposed secrets.
12. Deployment and remote changes occur only after explicit approval.

## 12. Risks and mitigations

- **Dataset provenance unclear:** inspect metadata and document limitations; avoid unsupported market-value claims.
- **Feature/target ambiguity:** confirm column meaning and units before final UI labels.
- **Data quality issues:** generate an audit report and make reproducible, conservative decisions.
- **Model overfitting:** use validation and leakage-safe preprocessing; optionally cross-validation when computationally reasonable.
- **Deployment artifact size or ephemeral storage:** assess model artifact size and deployment filesystem behavior before selecting a durable storage approach.
- **Security risks:** keep secrets server-side, validate requests, enforce ownership, and test CSRF/session behavior.
- **Insufficient data for some features:** defer unsupported functionality rather than fabricate records.

## 13. Open decisions to resolve during discovery

1. Confirm source URL/licence and the exact interpretation/units of `TARGET(PRICE_IN_LACS)`.
2. Confirm whether the application should be publicly accessible or limited to project/demo users.
3. Confirm exact feature fields and user-input constraints after dataset analysis.
4. Confirm email provider if password reset/email verification is required.
5. Confirm hosting environment variables, CORS origins, cookie settings, and deployment configuration before deployment.
6. Confirm whether a separate admin role is required for the assessed project.

## 14. Change control

Antigravity must read this PRD before implementation. If it discovers a conflict, missing requirement, or major architectural change, it must report the issue and ask for approval rather than silently changing scope.
