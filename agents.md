# AGENTS.md — EstateIQ Agent Instructions

**Project:** EstateIQ — An Intelligent Real Estate Price Prediction System Using Machine Learning  
**Applies to:** All AI agents and coding assistants working in this repository  
**Status:** Mandatory project-level instructions

## 1. First action: inspect and follow project requirements

Before planning or changing code:
1. Read `docs/PRD.md` and `docs/SRS.md` completely.
2. Inspect the current repository, existing files, Git status, dependency manifests, and relevant code before editing.
3. Identify conflicts, missing requirements, security risks, and assumptions. Ask the user when a decision materially affects architecture, data, security, cost, or scope.
4. Work in small, verifiable phases. Do not build or rewrite the entire application in one step.
5. Do not overwrite or delete existing files without inspecting them and explaining the planned change.

## 2. Absolute environment and secrets policy

### 2.1 Never access `.env` files

**Do not open, read, inspect, print, parse, copy, upload, summarize, or otherwise access the contents of any `.env` file**, including `.env`, `.env.local`, `.env.production`, `.env.development`, or other secret-bearing environment files.

- Do not use commands such as `cat`, `type`, `Get-Content`, `more`, `less`, `grep`, `findstr`, `Get-ChildItem` combined with content reads, Python scripts, IDE previews, or other tools to read `.env` contents.
- Do not ask the user to paste secrets or `.env` contents into chat.
- Do not include environment variable values in logs, screenshots, terminal output, generated reports, commits, pull requests, or documentation.
- Do not attempt to discover or extract secrets from local files, shell history, process listings, editor state, cloud configuration, or credential stores.
- You may check whether a required environment file exists only if needed, without opening it or revealing its contents. Prefer validating required variable *names* through code and reporting only whether configuration is missing or present, without echoing values.
- If a task appears to require reading an `.env` file, stop and ask the user to provide the required configuration through a secure, appropriate configuration workflow instead.

### 2.2 Safe configuration design

- Use environment-driven configuration for all environment-specific values and secrets.
- Create or maintain `.env.example` with variable names, descriptions, and clearly fake placeholders only. Never copy real values into it.
- Keep `.env`, `.env.*` secret-bearing files, private keys, credentials, tokens, and local secret files in `.gitignore`. If `.env.example` is intentionally tracked, explicitly allowlist it while continuing to ignore secret-bearing environment files.
- Load configuration through a validated settings module, such as Pydantic Settings for FastAPI, and use server-side environment variables for backend secrets.
- Frontend variables exposed to the browser (for example, `NEXT_PUBLIC_*`) must contain only values that are safe for public disclosure. Never place database passwords, session secrets, service-role keys, private API tokens, or other privileged credentials in frontend code or public environment variables.
- Never expose the Supabase service-role key or direct database credentials to the browser.
- Fail safely and clearly when required configuration is absent. Do not silently use insecure defaults or fake credentials.
- Do not print values when debugging configuration. Report only variable names and statuses such as “configured” / “missing.”
- Do not use a secret value as a fallback in source code, tests, Docker files, CI files, or deployment configuration.
- Use hosting-provider secret settings for deployment; do not commit production secrets.

## 3. No hardcoding: environment-driven application

The application must be **configuration-driven, not hardcoded**.

- Do not hardcode secrets, credentials, API keys, tokens, database URLs, session secrets, hostnames, ports, CORS origins, frontend/backend base URLs, storage buckets, email-provider settings, model artifact paths, or environment-specific values in application source code.
- Put environment-specific settings in environment variables and document their names in `.env.example`.
- Centralize configuration in typed settings/configuration modules. Validate required values on startup and provide safe defaults only for non-sensitive local-development settings where appropriate.
- Do not hardcode model predictions, evaluation metrics, dashboard totals, users, property listings, prediction history, model-selection outcomes, or fake fallback responses.
- Do not hardcode dataset schemas or feature lists before inspecting the actual supplied dataset. Once confirmed, define a versioned, documented feature schema and validate it explicitly.
- Do not hardcode a model name as “best.” Select it from actual, comparable validation results and record the selection rule and output.
- Avoid duplicated business rules across frontend and backend. The backend is authoritative for validation, authorization, prediction, and persistence.
- UI text, feature options, and units must reflect verified requirements/data; do not invent property fields or price units.
- Do not use magic numbers where a named constant or validated configuration setting improves clarity. Domain rules should be documented and tested rather than hidden in scattered literals.
- It is acceptable to use stable technical constants when justified (for example, an algorithm class or HTTP status code). Do not over-engineer every constant into an environment variable.

## 4. Strict data integrity: no fabricated product data

- Use only the actual dataset files supplied or explicitly approved by the user.
- Do not generate, invent, scrape, or download additional training data without explicit approval.
- Never use synthetic/mock/sample records as real application data, dashboard values, training data, prediction history, or displayed results.
- Do not fabricate ML metrics, training success, prediction values, source provenance, or model artifacts.
- If data, labels, source metadata, or a required input is missing, report the exact missing requirement and pause the dependent work.
- Preserve original data files unchanged. Put reproducible transformations in separate processed-data outputs and document them.
- Do not claim the dataset contains verified sale prices unless its provenance supports that statement.
- The unlabeled test dataset must not be used to calculate supervised evaluation metrics.
- Test-only fixtures, if needed, must be clearly isolated and must never be presented as real property data. If the user's no-dummy-data requirement could prohibit a proposed fixture, ask before creating it.

## 5. Machine learning requirements

- Inspect dataset columns, data types, target meaning, units, missingness, duplicates, outliers, suspicious values, and leakage risks before training.
- Build reproducible training and evaluation scripts. Do not claim training has run unless it actually executed successfully.
- Compare Linear Regression, Decision Tree Regressor, Random Forest Regressor, and KNN Regressor when suitable for the inspected data.
- Use a justified split or cross-validation strategy. Fit imputers, encoders, scalers, and all learned preprocessing on training partitions only.
- Keep preprocessing and the selected estimator together in a fitted pipeline for inference.
- Calculate genuine MAE, RMSE, and R² where appropriate. Do not describe regression metrics as “accuracy.”
- Select the best model using actual validation results and a documented metric-based rule. Never invent or preselect a winner.
- Save model artifacts and metadata only after successful training. Record the schema/version, data fingerprint, configuration, split strategy, seed, metrics, and timestamp as appropriate.
- If the model artifact is missing, corrupt, or incompatible, return a controlled error; never generate a fake prediction.

## 6. Architecture and technology boundaries

Use the approved stack unless a change is proposed and approved:
- Frontend: Next.js, TypeScript, Tailwind CSS.
- Backend: Python, FastAPI, Pydantic, Uvicorn.
- ML: Pandas, NumPy, Scikit-learn, Joblib.
- Database: Supabase PostgreSQL.
- ORM: SQLAlchemy 2.x.
- Migrations: Alembic.
- Testing: pytest, frontend linting and type checks.

Keep frontend, backend, training, inference, database migrations, and tests separated. The frontend must call the backend API; trusted inference and privileged database operations belong on the server.

## 7. Database and migrations

- Use SQLAlchemy ORM and Alembic migrations.
- Do not use destructive SQL, reset databases, drop tables, or apply migrations to a remote/production database without explicit user approval.
- Test migrations against a local/test database first.
- Do not expose database URLs, passwords, or privileged keys in code or browser bundles.
- Enforce authentication and ownership checks in backend queries; users must not be able to access another user's prediction history.
- Do not seed fake production users, property records, predictions, or metrics.

## 8. Security requirements

- Use Argon2 password hashing through a maintained library; never store plaintext passwords.
- Use secure server-managed sessions, high-entropy session identifiers, expiry, and logout invalidation.
- Set session cookies `HttpOnly`; use `Secure` in HTTPS production and an appropriate `SameSite` policy.
- Implement and test CSRF protection for cookie-authenticated state-changing requests.
- Apply rate limiting to sensitive authentication endpoints.
- Configure explicit CORS origins through validated configuration; do not use permissive wildcard credential settings.
- Enforce authorization on the backend.
- Do not log passwords, cookies, session tokens, access tokens, personal data unnecessarily, or secret values.
- Do not weaken security controls merely to make a local test pass; document and isolate any development-only configuration.

## 9. Coding and change-management rules

- Make the smallest change that correctly meets the requirement.
- Follow existing project conventions and avoid unnecessary dependencies.
- Use type hints, clear naming, focused functions, and meaningful error handling.
- Run relevant tests, linting, and type checks when feasible; report actual outcomes and blockers.
- Never claim a test, build, migration, model training, commit, push, or deployment succeeded unless it actually did.
- Do not run commands with destructive or externally visible effects without user approval.
- Do not commit, push, publish, deploy, incur cloud charges, or modify external services unless the user explicitly approves the specific action.
- Before a substantial change, explain what will change, why, and what files are affected.
- Keep documentation synchronized with implemented behavior. Do not document unimplemented functionality as completed.
- If a requirement is ambiguous or conflicts with these instructions, pause and ask the user rather than guessing.

## 10. Required repository files

Maintain these files as applicable:
- `AGENTS.md` — these instructions.
- `docs/PRD.md` — product requirements.
- `docs/SRS.md` — software requirements.
- `README.md` — setup and usage.
- `.env.example` — variable names and fake placeholders only.
- `.gitignore` — ignore secrets, local data, caches, build outputs, and local artifacts as appropriate.
- `docs/ARCHITECTURE.md` — architecture and data flow.
- `docs/ML_PIPELINE.md` — dataset, training, evaluation, and model artifact process.
- `docs/TEST_PLAN.md` — test strategy and acceptance checks.

## 11. Completion checklist

Before declaring a phase complete, confirm:
- [ ] PRD/SRS were followed or approved deviations documented.
- [ ] No `.env` file contents were accessed or exposed.
- [ ] No secrets or environment-specific values were hardcoded.
- [ ] No fabricated data, metrics, or prediction results were introduced.
- [ ] Relevant tests/checks were run and their actual outcomes reported.
- [ ] Documentation and configuration examples were updated.
- [ ] No unapproved remote, destructive, commit, push, or deployment action occurred.
- [ ] Any missing inputs, risks, and next steps are clearly reported.
