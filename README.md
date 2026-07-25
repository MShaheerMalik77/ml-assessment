# ML Assessment (Solvefy)

> Performed Exploratory Data Analysis (EDA) on Customer Support Ticket Dataset (Suraj520)


## Overview

In this iteration of the task, I performed EDA on the dataset, to understand its limitations and prepare it for model training.

## Features

- Missingness Checks
- Class Distribution
- Text Length Check
- Duplication Check
- Sanity Check
- Preprocessing
- Train Test Split

---

## Analysis

The EDA showed that the class distribution has a slight skew, which is important to note as a skewed distribution would negatively impact the model training. Up next I will use TF-IDF to vectorize and then compare Naive Bayes with Logistic Regression to identify which model performs with more accuracy. After this, if one model's F1 Macro is notably higher than accuracy I would need to handle minority classes better.

---