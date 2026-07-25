# ML Assessment (Solvefy)

> Performed Exploratory Data Analysis (EDA) and trained Multinomial Naive Bayes and Logistic Regression on AG News Dataset


## Overview

I initially attempted this task on a Kaggle Customer Support Ticket dataset, as it was on topic for query . Both Naive Bayes and Logistic Regression performed at around 18-20% on 5 balanced classes using ticket description and subject fields. Upon inspection of the raw text, I found unfilled template placeholders (e.g. {product_purchased}) and very similar generic phrasing reused across categories, showing that there is little context available to differentiate classes. I switched to AG News to deliver a working, evaluable pipeline.

## Features

- Missingness Checks
- Class Distribution
- Text Length Check
- Duplication Check
- Sanity Check
- Preprocessing
- Train Test Split
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Performance Metrics
---

##  Data Analysis

The EDA showed that the class distribution is well balanced. It is important to note this as a skewed distribution would negatively impact the model training.
---

## Findings

Logistic Regression Edges out Multinomial Naive Bayes slightly, reporting an 88.9% Accuracy over Naive Bayes' 87.1%. In this case I will proceed further using Logistic Regression and modularize the pipeline into seperate Python files for ease in reproduction. I will also add the API using FastAPI to serve the model. 

---

## Author
**Muhammad Shaheer Malik**