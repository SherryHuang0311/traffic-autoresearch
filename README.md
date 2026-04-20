
# Traffic Congestion Prediction in Chicago

## Overview
This project builds a reproducible baseline pipeline to predict whether a Chicago traffic segment will be congested 30 minutes into the future using historical traffic data.

The goal of Week 2 is to establish a stable and trustworthy evaluation pipeline before introducing automated experimentation.

## Research Question
Can we predict whether a Chicago traffic segment will be congested 30 minutes ahead using historical speed and time-based features?

## Dataset
This project uses a subset of the Chicago Traffic Tracker historical congestion dataset.

The dataset contains:
- timestamp (`TIME`)
- traffic segment ID (`SEGMENT_ID`)
- estimated speed (`SPEED`)
- hour of day (`HOUR`)
- day of week (`DAY_OF_WEEK`)

For reproducibility and manageable runtime, this project uses a fixed subset stored locally as:

`data/raw/traffic.csv`

## Baseline Model
The baseline model is logistic regression.

## Features
The baseline uses:
- current speed
- lag 1 speed
- lag 2 speed
- lag 3 speed
- hour of day
- day of week

## Target
The model predicts whether a traffic segment will be congested 30 minutes later.

Congestion is defined using the 30th percentile of training-set speed values.

## Data Split
The data split is deterministic and time-based:
- train: earliest 70% of timestamps
- validation: next 15%
- test: final 15%

The final test set is locked and is not used during development.

## Evaluation Metric
The fixed validation metric is F1 score.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt

