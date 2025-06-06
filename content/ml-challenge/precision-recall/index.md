---
title: "Precision Recall and other Classification Metrics"
date: 2025-06-06T17:36:17+09:00
author: "Lukas Hofbauer"
github: "https://github.com/itsfernn/ml-daily-challenge/blob/main/precision-recall/precision-recall.ipynb"
cover:
    image: "cover.png"
---

When evaluating a classification model accuracy alone isn’t enough. To better understand how well your model is performing, we need to dig deeper by understanding metrics like **precision**, **recall**, **F1 score**, and performance curves like **ROC** and **Precision-Recall (PR)**.

We'll start by using the same classifier as in the [Logistic
Regression](logistic-regression) post.


```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris["data"][:,1].reshape(-1,1)
y = (iris["target"] == 0).astype(int)

log_reg = LogisticRegression()
log_reg.fit(X,y)
```


    
![png](output_1_0.png)
    


## The Confusion Matrix

Everything starts with the **confusion matrix**, which keeps track of four outcomes in binary classification:

* **True Positives (TP)**: predicted positive, actually positive
* **False Positives (FP)**: predicted positive, actually negative
* **False Negatives (FN)**: predicted negative, actually positive
* **True Negatives (TN)**: predicted negative, actually negative

From these four numbers, we can define our most important metrics.


```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = log_reg.predict(X)

cm = confusion_matrix(y, y_pred)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Virginica", "Virginica"])
fig, ax = plt.subplots()
disp.plot(cmap='Blues', ax=ax)
```


    
![png](output_3_0.png)
    


# Precision and Recall
**Precision** answers the question: *Out of all the instances the model predicted as positive, how many were actually positive?*

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

For example, in a spam filter, precision tells you how many emails flagged as spam were truly spam.

However, precision alone can be misleading. A model that predicts "spam" just once—and gets it right—but always says "not spam" after that, can still have perfect precision. But it's clearly useless!

---

That’s where **recall** comes in. Also known as **sensitivity** or **true positive rate**, it answers: *Out of all actual positive cases, how many did the model catch?*

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

In the spam example, recall measures how many actual spam emails were successfully identified.



```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
```

    Precision: 0.838
    Recall:    0.620


## F1 Score

So both Precision and recall are important but it would be nice to combine them into a single metric. For this we use the **F1 score**; the **harmonic mean** of precision and recall:

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Why harmonic mean? Because it punishes imbalance: if either precision or recall is low, the F1 score drops significantly. This makes F1 a great metric when you care about **both** catching positives and avoiding false alarms.



```python
from sklearn.metrics import f1_score

f1 = f1_score(y, y_pred)
print(f"F1 Score:  {f1:.3f}")
```

    F1 Score:  0.713


## The Precision–Recall Trade-Off

Most classifiers give a score or probability for each instance. To turn that into a decision, you pick a **threshold** (e.g., 0.5). Adjusting this threshold shifts the balance between precision and recall:
- Increase the threshold = *higher precision* but *lower recall*
- Decrease the threshold = *higher recall* but *lower precision*

This trade-off is **context-dependent**:
- In a **spam filter**, you'd want **high precision** — it's okay to miss some spam, but misclassifying a real email as spam (false positive) is bad.
- In **medical diagnosis**, you want **high recall** — better to flag a potential issue and follow up than to miss a real case.

---

## Precision-Recall Curve and PR-AUC

To visualize the trade-off, we plot the **Precision-Recall (PR) curve**:

* **x-axis**: Recall
* **y-axis**: Precision

You can trace this curve by varying the classification threshold from 0 to 1.

To summarize the entire curve with a single number, we use the **PR AUC**—the **area under the precision-recall curve**. A higher PR AUC indicates better performance across all thresholds, especially when **positive cases are rare**.



```python
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities for the positive class (Virginica)
y_scores = log_reg.predict_proba(X)[:, 1]

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, y_scores)

# Compute PR AUC
pr_auc = auc(recall, precision)

# Plot PR curve and fill area under it
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, color="tab:blue", label="Precision-Recall Curve (PR-Curve)")
plt.fill_between(recall, precision, alpha=0.2, color="tab:blue", label=f"Area under the Curve (AUC) = {pr_auc:.3f}")
```

    
![png](output_9_0.png)
    


## ROC Curve and ROC AUC

Another common performance curve is the **ROC curve (Receiver Operating Characteristic)**. It plots:

* **x-axis**: False Positive Rate = FP / (FP + TN)
* **y-axis**: True Positive Rate = Recall = TP / (TP + FN)

A **random model** gives you a diagonal line (AUC = 0.5), while a **perfect classifier** reaches the top-left corner (AUC = 1.0).

$$
\text{ROC AUC} = \text{Area under the ROC curve}
$$

ROC AUC is especially useful when **positive and negative classes are balanced**. It measures the model’s ability to **rank** positive instances higher than negatives, regardless of the threshold.



```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute precision-recall curve
tpr, fpr, thresholds = roc_curve(y, y_scores)

# Compute PR AUC
pr_auc = auc(tpr, fpr)

# Plot PR curve and fill area under it
plt.figure(figsize=(8, 5))
plt.plot(tpr, fpr, color="tab:green", label="ROC Curve")
plt.fill_between(tpr, fpr, alpha=0.2, color="tab:green", label=f"Area under the Curve (AUC) = {pr_auc:.3f}")
```

    
![png](output_11_0.png)
    


## PR AUC vs. ROC AUC

To summarize, use the Precision-Recall (PR) curve when your classes are imbalanced or you care more about the positive class, as it focuses on precision and recall. This might necessitate switching which class is the positive class (spam or ham?). The ROC curve works well when classes are balanced since it treats both classes equally. In short, PR curves highlight positive-class performance, while ROC curves measure overall discrimination.
