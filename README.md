# Chicago Crime Risk Score, MLOps Final Project

SISTECH 2026, Machine Learning Operations path
Greycia Febrina Michelle & Aulia Aca Azzahra

## Overview

This project builds an end to end Risk Score prediction system for a location and time, based on historical Chicago crime data. It continues directly from Hands-On 1 (pseudo-labeling and feature engineering) and Hands-On 2 (modeling and continual learning), now combined into one system with a model, a REST API, and monitoring.

Risk Score reflects three dimensions at once: how severe past crimes were, how recently they happened, and how close they were to the location being asked about.

Disclaimer: this system produces an estimate based on historical patterns, not a certainty that anything will happen. The dataset is from Chicago, not Indonesia, and is used here to validate the modeling approach until a local dataset becomes available.

## Project Status

- CP1, Pseudo-Labeling and Feature Engineering: done
- CP2, Model Training, Baseline Comparison, REST API Serving: in progress
- CP3, Continual Learning, Monitoring and Logging, Documentation: not started yet

## Repository Structure

| Path | Description |
|---|---|
| `notebooks/` | Step by step notebooks for each stage of the project |
| `src/` | Shared logic reused by both notebooks and the API, so scoring behaves identically in training and serving |
| `app/` | The FastAPI application, added in CP2 |
| `models/` | Saved model checkpoints and the version registry, added in CP2 and CP3 |
| `logs/` | Prediction activity logs, added in CP2 |
| `data/` | Instructions for obtaining the raw dataset |

## How to Run the Notebooks

1. Get `events_scored.csv` following the instructions in `data/README.md`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/01_pseudo_labeling_feature_engineering.ipynb` and run all cells in order

## How to Run the API

Not available yet, this section will be filled in once CP2 is complete.
