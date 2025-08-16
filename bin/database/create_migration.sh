#!/bin/bash
# Script to create initial migration

echo "Creating initial database migration..."

# Generate migration
alembic revision --autogenerate -m "Initial migration with TimescaleDB tables"

echo "Migration created. Review the generated file in alembic/versions/"
echo "To apply the migration, run: make migrate"