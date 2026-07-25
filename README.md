# Customer Query Classification — ML Assessment (Solvefy)

> Trains a TF-IDF + Logistic Regression classifier on the AG News dataset and 
> serves predictions via a FastAPI `/predict` endpoint.

## Overview

I initially attempted this task on a Kaggle Customer Support Ticket dataset, 
since it's directly on-topic for query routing. Both Naive Bayes and Logistic
Regression performed at around 18-20% on 5 balanced classes using ticket 
description and subject fields. Inspection of the raw text showed the presence of 
unfilled template placeholders (e.g. `{product_purchased}`) and very similar 
phrasing reused across categories, indicating that the text did not carry significant
class differentation signal. This investigation is preserved in `experiment_01/`.
I switched to the AG News dataset to deliver a working, evaluable pipeline 
within the available time.

## Setup Instructions

1. Clone the repo and install dependencies:
```bash
   git clone https://github.com/MShaheerMalik77/ml-assessment.git
   cd ml-assessment
   pip install -r requirements.txt
```

2. Download the AG News dataset (`train.csv`, `test.csv`) and place them in `data/`.
   [Link to dataset source]

3. Train the model:
```bash
   python src/train.py
```
   This saves the fitted pipeline to `models/model.pkl` and writes evaluation 
   output to `reports/metrics.txt` and `reports/confusion_matrix.png`.

4. Start the API:
```bash
   cd src
   uvicorn api:app --reload --port 8000
```

5. Test it — either open `http://localhost:8000/docs` for the interactive UI, or:
```bash
   python test_api.py
```
   Example request:
```json
   POST /predict
   {"text": "The stock market rallied today after the Fed announcement"}
```
   Example response:
```json
   {"category": "Business", "confidence": 0.90}
```

## Methodology

**Preprocessing:** lowercasing, URL stripping, non alphanumeric removal,
whitespace normalization, and stopword removal, were all applied identically at 
both training and inference time via the `preprocessing.py` module. As such I  
avoided train/serve skew.

**Vectorization:** I chose TF-IDF over raw bag of words counts because it 
downweights terms that are common across all categories (such as generic words that 
survive stopword removal) and upweights terms that are more distinctive to a 
given class. This results in a better fit than counts alone for a routing task
where distinguishing vocabulary matters more than raw frequency.

**Model:** Logistic Regression and Multinomial Naive Bayes were both trained 
and compared on identical TF-IDF features. Naive Bayes is a common, fast 
baseline for text classification, however it assumes independence of words, which 
is a poor fit for natural language, where word co-occurrence has meaning. 
Logistic Regression on the other hand does not make this assumption and weighs
combinations of features in a more flexible manner. This is reflected in its 
slightly stronger result mentioned below. Both models are lightweight and train 
quickly, along with being easy to serve. This satisfies the requirement of lightweight
and reliable as opposed to a deep learning model that would add latency and infrastructure
cost.

## Evaluation

| Model | Accuracy | Macro F1 |
|---|---|---|
| Multinomial Naive Bayes | 87.1% | 87.1% |
| Logistic Regression | 88.9% | 88.9% |

Logistic Regression was selected as the final model.

**Metric choice:** I reported accuracy which is reliable since AG News' 4 classes are
evenly balanced (confirmed during EDA).  F1 Macro is also listed alongside as it 
weighs each class equally, regardless of support. This is the most robust metric if
class balance shifts during production. For a query routing system, misclassifying a
minority department's queries is just as costly as misclassifying a majority one, so
macro F1 is more relevant in this scenario though both are close here due to the proper 
class balance. A confusion matrix (`reports/confusion_matrix.png`) was also included to 
check any errors in specific class pairs which share vocabulary.


## Project Structure
```
├── data/ # AG News train/test CSVs 
├── experiments/ # Customer-ticket investigation (see Overview)
├── src/
│ ├── preprocessing.py # Shared text cleaning, used by train.py and api.py
│ ├── train.py # Trains and saves the pipeline
│ ├── api.py # FastAPI serving layer
│ └── models/
│   └── model.pkl # Fitted TF-IDF + LogisticRegression pipeline
├── reports/
│ ├── metrics.txt
│ └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## Author
**Muhammad Shaheer Malik**
