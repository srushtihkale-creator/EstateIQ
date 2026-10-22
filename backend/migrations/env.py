"""Alembic Environment Script for EstateIQ Database Migrations."""

from logging.config import fileConfig
import os
import sys
from pathlib import Path
from sqlalchemy import engine_from_config, pool

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from backend.app.db.session import Base
# Import all models so that Alembic detects them in Base.metadata
from backend.app.models.user import User
from backend.app.models.session import UserSession
from backend.app.models.prediction import PredictionHistory

# this is the Alembic Config object
try:
    from alembic import context
    config = context.config

    if config.config_file_name is not None:
        fileConfig(config.config_file_name)

    target_metadata = Base.metadata

    def run_migrations_offline() -> None:
        """Run migrations in 'offline' mode."""
        url = settings.DATABASE_URL
        context.configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )

        with context.begin_transaction():
            context.run_migrations()

    def run_migrations_online() -> None:
        """Run migrations in 'online' mode."""
        configuration = config.get_section(config.config_ini_section) or {}
        configuration["sqlalchemy.url"] = settings.DATABASE_URL

        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
            )

            with context.begin_transaction():
                context.run_migrations()

    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()

except ImportError:
    # Graceful handling when alembic package is not yet installed in active environment
    pass
