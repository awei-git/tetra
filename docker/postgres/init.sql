-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS events;
CREATE SCHEMA IF NOT EXISTS derived;
CREATE SCHEMA IF NOT EXISTS strategies;
CREATE SCHEMA IF NOT EXISTS execution;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA market_data TO tetra_user;
GRANT ALL PRIVILEGES ON SCHEMA events TO tetra_user;
GRANT ALL PRIVILEGES ON SCHEMA derived TO tetra_user;
GRANT ALL PRIVILEGES ON SCHEMA strategies TO tetra_user;
GRANT ALL PRIVILEGES ON SCHEMA execution TO tetra_user;

-- Set default search path
ALTER DATABASE tetra SET search_path TO public, market_data, events, derived, strategies, execution;

-- Create tables will be handled by Alembic migrations
-- This file just sets up the initial database structure