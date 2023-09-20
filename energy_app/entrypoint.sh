#!/bin/sh

# Update the database schema
alembic upgrade head

# Assure the database countries table is populated & up-to-date
python load_db_fixtures.py

exec "$@"