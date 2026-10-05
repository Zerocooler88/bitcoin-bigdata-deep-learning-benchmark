# Reproducible Big Data Benchmark for Bitcoin Price Forecasting

This repository contains the reproducibility resources associated with the study:

**A reproducible Big Data benchmark for Bitcoin price forecasting using deep learning models and a persistence baseline**

## Overview

The benchmark compares a persistence baseline with LSTM, GRU, CNN-1D, and CNN-LSTM models for one-step-ahead BTCUSDT Spot price forecasting.

The study emphasizes reproducibility through a Bronze/Silver/Gold Big Data architecture, strict chronological evaluation, multi-seed experiments, scenario-based stress testing, statistical comparison with persistence, and sensitivity analyses.

## Main technologies

- Apache Spark
- MinIO
- Apache Airflow
- Docker
- PyTorch
- CUDA

## Repository structure

- `src/`: data processing, forecasting, statistical analysis, and figure-generation scripts.
- `logreturn_sensitivity/`: derived results from the log-return sensitivity experiment.
- `requirements_q1_base.txt`: core Python dependencies.
- `requirements_q1_frozen.txt`: frozen reproducibility environment.
- `.env.example`: example environment configuration without real credentials.

## Data

The study uses publicly available BTCUSDT Spot market data obtained from Binance.

Large Bronze, Silver, Gold, and Parquet datasets are not distributed through GitHub because of their size. The provided processing scripts support reconstruction of the analytical datasets.

## Credentials

Real credentials are not stored in this repository. MinIO credentials must be supplied through environment variables. See `.env.example`.

## Main analyses

The repository includes scripts for:

- persistence-baseline evaluation;
- LSTM, GRU, CNN-1D, and CNN-LSTM experiments;
- strict chronological and hourly-continuity controls;
- five-seed reproducibility analysis;
- 13-scenario stress testing;
- HAC/Newey-West predictive-accuracy sensitivity analysis;
- hierarchical block-bootstrap RMSE-ratio inference;
- Mincer-Zarnowitz residual-correction analysis;
- log-return sensitivity analysis;
- scientific figure generation.

## Archival release

The version associated with the manuscript will be archived in Zenodo.

**Zenodo DOI:** to be added after publication of the archival release.

## Disclaimer

This repository is provided for scientific research and reproducibility purposes. The forecasting experiments do not constitute financial or investment advice.
