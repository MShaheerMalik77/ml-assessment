# ML Assessment (Solvefy)

> Performed Exploratory Data Analysis (EDA) and trained Multinomial Naive Bayes and Logistic Regression on Customer Support Ticket Dataset (Suraj520)


## Overview

While working on this assessment, I performed EDA on the dataset, to understand its limitations and prepare it for model training, then, after TF-IDF vectorization I trained Multinomial Naive Bayes and Logistic Regression Models on the processed data.

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

The EDA showed that the class distribution is mostly balanced, which is important to note as a skewed distribution would negatively impact the model training.
---

## Findings

Both models post-training achieved a mere 20% accuracy, despite relatively balanced classes. Upon inspection, the output of Ticket Types corresponding to Ticket Description shows that the description for most tickets is very similar. As the dataset was synthetically generated,  the raw text had heavy template reuse and placeholder tokens across categories, which shows that ticket description does not serve as an accurate indicator of a class. I confirmed this wasn't a preprocessing or leakage bug by checking X. After replacing ticket description with ticket subject, both models reported an accuracy of 18%, showcasing that the problem does indeed lie with the dataset. Feature Engineering would result in a data type that contains the same lable-text mismatch, therefore I will be making use of the AG News dataset for classification.

---

## Author
**Muhammad Shaheer Malik**