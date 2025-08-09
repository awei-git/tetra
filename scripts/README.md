# Scripts Directory

This directory is for **temporary and experimental scripts only**.

## ⚠️ IMPORTANT: This folder is regularly cleaned

All files in this directory (except this README) may be deleted at any time without notice.

## ❌ What does NOT belong here

- Production scripts → Use `bin/` instead
- Database migrations → Use `backend/alembic/versions/`
- Pipeline code → Use `src/pipelines/`
- Any script that other code depends on → Move to appropriate `src/` module

## ✅ What belongs here

- Quick test scripts for debugging
- One-off data exploration scripts
- Temporary fixes or patches
- Experimental code that will be deleted
- Scripts for testing new ideas

## 📁 Where to put production code

- `bin/` - Operational scripts (startup, pipelines, monitoring)
- `src/` - Production Python modules and packages
- `backend/` - Backend API specific code
- `config/` - Configuration files

## 🗄️ Archive

Previously used scripts are archived in `archive/scripts_backup/` for historical reference only.

## 🧹 Last cleaned: August 2025

All experimental scripts were removed as they had no dependencies.