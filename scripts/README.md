# Scripts Directory

This directory is for **temporary and experimental scripts only**.

## Important Notes

- Do NOT put production scripts here
- All operational scripts should go in `bin/`
- Database migrations go in `backend/alembic/versions/`
- Pipeline code goes in `src/pipelines/`

## What belongs here

- Quick test scripts for debugging
- One-off data exploration scripts
- Temporary fixes or patches
- Experimental code that will be deleted

## Cleanup Policy

Files in this directory may be deleted at any time without notice.
If you have a script that needs to be preserved, move it to the appropriate location:

- `bin/` - Operational scripts (startup, monitoring, etc.)
- `src/` - Production Python code
- `backend/` - Backend-specific scripts
- `config/` - Configuration files

## Archive

A backup of previous scripts is kept in `archive/scripts_backup/` for reference.