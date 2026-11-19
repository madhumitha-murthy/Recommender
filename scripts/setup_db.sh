#!/bin/bash
# Create the PostgreSQL database for the recommender project.
# Assumes PostgreSQL is running locally.
# Usage: bash scripts/setup_db.sh

set -e

DB_NAME="nbp_recommender"
DB_USER="${PGUSER:-postgres}"

echo "Creating database '$DB_NAME'..."
psql -U "$DB_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || true
psql -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"
echo "Database '$DB_NAME' created."

echo "Running schema..."
psql -U "$DB_USER" -d "$DB_NAME" -f src/data/schema.sql
echo "Schema applied."

echo "Done. Run 'python src/data/load_to_postgres.py' to load data."
