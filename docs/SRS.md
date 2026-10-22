# EstateIQ — Software Requirements Specification (SRS)

**Document version:** 1.0  
**Status:** Initial technical baseline for review  
**Project:** EstateIQ — An Intelligent Real Estate Price Prediction System Using Machine Learning

## 1. Purpose

This SRS specifies the technical and behavioral requirements for implementing EstateIQ. It is intended for Antigravity-assisted development, implementation review, testing, and deployment planning. Read it together with `PRD.md`.

## 2. System context and architecture

The system consists of:
1. **Next.js frontend** — renders pages and forms; calls the backend API; contains no secret keys and does not run trusted ML inference in the browser.
2. **FastAPI backend** — validates requests, manages authentication/session behavior, invokes inference, and enforces user-level access control.
3. **ML training pipeline** — inspects data, fits preprocessing and candidate regressors, evaluates them, and emits a versioned model artifact and report.
4. **Inference package** — loads a verified artifact and exposes a stable prediction interface.
5. **Supabase PostgreSQL** — stores application accounts/session records as designed, prediction history, and model metadata if needed.
6. **Alembic migrations** — version and apply schema changes.
7. **Test and deployment environments** — isolated from production by configuration.

### 2.1 Request flow
Browser → Next.js frontend → FastAPI endpoint → request validation → inference service → fitted preprocessing/model pipeline → prediction response → optional authenticated history persistence → frontend result.

### 2.2 Training flow
User-provided dataset → data audit → split strategy → training-only preprocessing fit → candidate model training → validation evaluation → selection → save fitted full pipeline and metadata/report → inference loads artifact.

## 3. Runtime and dependency requirements

- Use a currently supported Python version compatible with selected FastAPI, SQLAlchemy, Scikit-learn, and deployment runtime; pin tested dependencies.
- Use a supported Node.js LTS version compatible with the selected Next.js release; pin package dependencies through the package lockfile.
- Use a supported PostgreSQL version supplied by Supabase.
- Keep frontend and backend dependency manifests separate.
- Record exact tested versions in documentation after installing and verifying them; do not claim versions are tested until they are.
- Use `.env.example` with placeholder variable names only; never include real secrets.
- Provide a root README with setup commands and separate service-specific instructions.

## 4. Data requirements

### 4.1 Current dataset expectations
The supplied archive contains:
- `train.csv`: labeled training data with `TARGET(PRICE_IN_LACS)` as the apparent target column.
- `test.csv`: feature data without the target.
- `sample_submission.csv`: submission template.

These names and the target's meaning must be confirmed by the data-inspection code and available source metadata before training.

### 4.2 Data directory rules
- `ml/training/data/raw/`: original files, kept unchanged.
- `ml/training/data/processed/`: reproducible derived files.
- `ml/training/data/external/`: only separately approved external source data, if any.
- Do not commit private or large dataset files to Git unless the user explicitly approves and repository policy permits it.
- Do not automatically upload the dataset to Supabase Storage.

### 4.3 Data audit
Create a repeatable audit that reports:
- file presence, shape, columns, data types, and target presence;
- missing values and duplicate rows;
- categorical/numeric distributions;
- target and numeric feature distributions;
- potential outliers and invalid ranges;
- coordinate plausibility if coordinates are used;
- leakage risks and ID-like columns;
- source/provenance assumptions and unresolved questions.

Do not silently remove records or impute arbitrary values. Document every cleaning operation and preserve original data.

## 5. ML pipeline requirements

### 5.1 Candidate models
Implement and compare, when compatible with data size and feature schema:
1. `LinearRegression`
2. `DecisionTreeRegressor`
3. `RandomForestRegressor`
4. `KNeighborsRegressor`

Additional algorithms require a documented reason. Candidate models must use comparable data splits and eligible features. Computationally expensive settings may be tuned within practical limits and documented.

### 5.2 Split and leakage prevention
- Separate target from features before transformations.
- Keep a holdout validation set or use a justified cross-validation strategy.
- If repeated rows, temporal structure, groups, or source ordering make random splitting inappropriate, document and use a suitable strategy.
- Fit imputers, encoders, scalers, feature transformations, and feature selection on training data only.
- Use a Scikit-learn `Pipeline` and `ColumnTransformer` where suitable so training and inference share identical transformations.
- Do not use the unlabeled test CSV to calculate supervised metrics or to tune decisions based on nonexistent labels.
- Fix and record random seeds where applicable.

### 5.3 Evaluation and selection
- Compute MAE and RMSE; compute R² where appropriate and explain its interpretation.
- Report sample/split sizes and the exact validation strategy.
- Do not call regression metrics “accuracy.”
- Choose a primary selection metric before final selection. Default proposal: minimize validation RMSE, using MAE and R² as complementary measures; document any justified alternative and ask for approval if the selection policy materially changes.
- If validation results are inconclusive or the data is not adequate, state that limitation rather than inventing a winner.
- Metrics must be generated by executed evaluation code and saved in machine-readable JSON plus a human-readable report.

### 5.4 Artifact requirements
- Save the fitted preprocessing + selected estimator as one versioned Joblib pipeline artifact after successful training.
- Save metadata including model class, feature names/schema, target name/units assumption, preprocessing version, training time, random seed, dataset fingerprint or source file hashes, validation strategy, and genuine metrics.
- Never load untrusted user-supplied Joblib files.
- Do not check large trained artifacts into Git without approval; decide deployment packaging after checking artifact size and host persistence.
- Inference must fail safely with a clear service-unavailable response if the artifact is missing, invalid, or incompatible.

### 5.5 Inference schema
- Define request and response schemas with Pydantic.
- Build the request feature schema only after inspecting actual dataset features and deciding which are appropriate for user input.
- Reject missing required fields, invalid types, non-finite numbers, and invalid values.
- Do not silently replace unsupported or missing input fields with invented defaults.
- Return the numeric estimate, target unit/scale metadata, model version, and a non-guarantee disclaimer as appropriate.
- Do not expose local filesystem paths or internal exception traces in public API responses.

## 6. Backend requirements

### 6.1 Suggested package responsibilities
- `app/api/`: routers and endpoint definitions.
- `app/core/`: configuration, security utilities, logging, and shared exceptions.
- `app/models/`: SQLAlchemy ORM models.
- `app/schemas/`: Pydantic request/response schemas.
- `app/services/`: business logic, authentication, prediction, and history operations.
- `app/db/`: engine/session setup and database utilities.
- `migrations/`: Alembic environment and versioned migration scripts.
- `tests/`: unit, integration, and API tests.

### 6.2 Minimum API surface
Exact route names may be finalized during implementation, but provide equivalent documented routes:
- `GET /health` — process health; do not leak environment secrets.
- `GET /ready` — readiness including required model artifact/configuration state.
- `POST /api/auth/register` — validate and create account.
- `POST /api/auth/login` — verify credentials and establish session.
- `POST /api/auth/logout` — invalidate session.
- `GET /api/auth/me` — return current user's safe profile.
- `POST /api/predictions` — validate inputs, run inference, optionally persist result for authenticated user.
- `GET /api/predictions` — return only the authenticated user's history, paginated.
- `GET /api/predictions/{prediction_id}` — return a record only if it belongs to the authenticated user or the requester is explicitly authorized.
- `GET /api/model/metrics` — return genuine saved evaluation metadata only; do not trigger expensive training on a public request.

Return suitable HTTP status codes and a consistent error shape. Document request and response examples using non-fabricated schemas; any illustrative examples must be clearly labeled as examples and not presented as real prediction output.

## 7. Authentication and security requirements

- Use Argon2 password hashing through a maintained library; never store plaintext passwords or reversible password values.
- Use server-managed sessions with random, high-entropy session identifiers. Store only a safe hash/representation of session tokens if practical.
- Session cookies must be `HttpOnly`, `Secure` in HTTPS production, and use an appropriate `SameSite` policy.
- Implement CSRF protection for cookie-authenticated state-changing requests; define and test the token/origin strategy.
- Enforce session expiration and logout invalidation.
- Rate-limit login and registration attempts; avoid revealing whether an account exists in sensitive error messages.
- Validate `Origin`/CORS configuration against explicit frontend origins; do not use permissive wildcard credentials.
- Enforce authorization and prediction ownership on the backend, not only in the UI.
- Use environment variables for secrets, database URLs, session keys, and provider credentials.
- Never expose the Supabase service-role key, database password, or backend secret in frontend environment variables, logs, screenshots, or committed files.
- Add secure headers and safe error handling where appropriate.
- If password reset or email verification is implemented, configure a real delivery provider and token expiry/one-time use. Do not claim messages were sent unless delivery is integrated.
- Do not run migrations or destructive operations against remote/production data without explicit approval.

## 8. Database and ORM requirements

- Use SQLAlchemy 2.x ORM with explicit relationships, constraints, indexes, and typed mappings where appropriate.
- Use Alembic migrations for schema evolution; do not rely on `create_all()` as the production migration mechanism.
- Keep database sessions scoped correctly and closed reliably.
- Store only necessary account/profile data; never store plaintext passwords.
- Prediction-history records should include the owning user ID, input snapshot, prediction value, timestamp, model/artifact version, and relevant schema metadata.
- Enforce foreign keys and appropriate deletion behavior; consider user privacy and retention.
- Add pagination and safe query limits for history.
- Keep model metrics in a versioned artifact/report or an appropriate metadata table; do not fabricate seeded records.
- Use a separate local/test database for tests. Remote migration requires explicit approval.

## 9. Frontend requirements

- Use Next.js with TypeScript and Tailwind CSS.
- Organize reusable components, typed API client, page routes, and form validation.
- Use React Hook Form and Zod if useful for the final feature schema.
- Do not duplicate backend business rules as the sole source of validation; backend validation is authoritative.
- Provide loading, success, empty, unauthorized, validation, and server-error states.
- Do not store privileged secrets in `NEXT_PUBLIC_*` variables.
- Use charts only when genuine model metrics or actual user data are available.
- Make pages responsive and accessible with keyboard support and semantic labels.
- Do not invent property listings or dashboard counts to fill empty states.

## 10. Testing requirements

### 10.1 ML tests
- Audit detects missing required files/target columns.
- Preprocessing fit is confined to training data.
- Pipeline accepts valid schema and rejects invalid/missing/non-finite input.
- Metrics are computed from real evaluation labels.
- Saved artifact can be loaded and used for inference.
- Missing/corrupt artifact produces a controlled failure.
- Test data without labels is not used for supervised scoring.

### 10.2 Backend tests
- Health/readiness behavior.
- Authentication registration/login/logout/session expiry.
- Password hashes are not plaintext.
- Invalid payloads receive appropriate errors.
- Prediction endpoint requires a valid model.
- User A cannot access User B's prediction history.
- Pagination, database errors, and authorization behavior.
- CSRF and rate-limit behavior where implemented.

### 10.3 Frontend tests and checks
- Lint and TypeScript checks pass.
- Forms display validation errors and loading/error states.
- API failures are handled.
- Protected pages require authentication.
- Prediction result and metrics views use API-provided values rather than hardcoded values.
- Responsive and basic accessibility checks.

### 10.4 Test-data policy
Tests may use minimal synthetic fixtures only when strictly necessary to test software behavior, must be clearly isolated under test fixtures, and must never be used for model training, product dashboards, real prediction examples, or displayed results. If the project requirement is interpreted as prohibiting even test-only fixtures, ask the user before creating them. Never disguise fixtures as real data.

## 11. Logging and observability

- Use structured logs where practical.
- Do not log passwords, session tokens, full secrets, or sensitive personal data.
- Log training run metadata and failure causes without exposing secrets.
- Keep a record of actual training run configuration and metrics.
- Expose health/readiness endpoints without leaking internal details.

## 12. Environment configuration

Provide `.env.example` with variable names and explanatory placeholders, for example:
- `DATABASE_URL`
- `SESSION_SECRET`
- `FRONTEND_ORIGIN`
- `MODEL_ARTIFACT_PATH`
- `MODEL_METADATA_PATH`
- `ENVIRONMENT`

Add optional email-provider variables only if email features are implemented. Do not place actual values in `.env.example`. Ensure `.env`, credentials, private datasets (if applicable), caches, and build artifacts are covered by `.gitignore`.

## 13. Migration, release, and deployment requirements

- Build and test locally before deployment.
- Run migrations against a local/test database first.
- Provide a documented migration review and rollback strategy.
- Deploy frontend to Vercel and backend to Render only after configuration and approval.
- Configure CORS, HTTPS, secure cookies, environment variables, health checks, and model artifact availability for production.
- Verify whether the backend filesystem is persistent; use deployment packaging or durable storage appropriate to the artifact size and host behavior.
- Do not automatically publish code, create cloud resources, incur charges, or modify production data.
- Record deployment status only after checking actual provider results.

## 14. Repository and code quality requirements

- Keep frontend, backend, and ML dependencies appropriately separated.
- Use formatting, linting, type checking, and tests.
- Avoid unnecessary dependencies and duplicated business logic.
- Use clear names, type hints, and meaningful error messages.
- Add docstrings/comments for non-obvious logic.
- Keep secrets out of source control.
- Do not silently overwrite existing files or restructure the repository without review.

## 15. Required deliverables

1. `README.md` with project overview and setup.
2. `docs/PRD.md` and `docs/SRS.md`.
3. Dataset audit report and reproducible cleaning/preprocessing code.
4. ML training and evaluation scripts.
5. Actual metrics report and versioned fitted pipeline after a successful run.
6. FastAPI backend with documented schemas/routes.
7. SQLAlchemy ORM models and Alembic migrations.
8. Next.js frontend integrated with the backend.
9. Automated tests and actual test-run report.
10. `.env.example` and `.gitignore`.
11. Architecture and ML pipeline documentation.
12. Deployment instructions and checklist.

## 16. Definition of done

A feature is complete only when its implementation exists, its relevant checks have been run, the results are reported accurately, and any remaining limitations are documented. Never claim a model was trained, a test passed, a migration was applied, or deployment succeeded unless evidence confirms it.

## 17. Open issues and approval gates

Before finalizing user-facing input fields and price units, confirm the dataset source, target meaning, and units. Before deploying, confirm the hosting configuration and security settings. Ask the user before adding a new data source, expanding product scope, running remote migrations, or committing/pushing/deploying changes.
