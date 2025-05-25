# Model Comparison

> **Заметка:** Бейзлайн — постоянное предсказание класса 1.

---

## 1. Baseline

| Model    | Features | Hyperparameters | Accuracy | F1 Score |
| -------- | -------- | --------------- | -------- | -------- |
| Baseline | N/A      | N/A             | 0.508    | 0.674    |

---

## 2. Classical Machine Learning

### 2.1 Linear Models

| Model               | Features               | Hyperparameters                                                                  | Accuracy | F1 Score |
| ------------------- | ---------------------- | -------------------------------------------------------------------------------- | -------- | -------- |
| Logistic Regression | One-hot encoding (OHE) | **C=1.0**, penalty=`l2`, solver=`lbfgs`, class\_weight=`None`                    | 0.586    | 0.590    |
| Logistic Regression | OHE                    | **C=0.0001**, penalty=`l2`, solver=`newton-cg`, class\_weight=`balanced`         | 0.598    | 0.604    |
| Linear SVC          | OHE                    | **C=1.0**, penalty=`l2`, class\_weight=`None`                                    | 0.593    | 0.596    |
| SVC (poly kernel)   | OHE                    | **C=0.001**, kernel=`poly`, degree=5, shrinking=`True`, class\_weight=`balanced` | 0.519    | 0.672    |

### 2.2 Gradient-Boosted Trees

| Model                        | Features                       | Hyperparameters                                                                                                                               | Accuracy | F1 Score |
| ---------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- |
| **GBC (no embeddings)**      | OHE                            | n\_estimators=460, learning\_rate=0.1027, max\_depth=9, min\_samples\_split=10, min\_samples\_leaf=4, subsample=0.9996, max\_features=`None`2 | 0.765    | 0.768    |
| CatBoost (no embeddings)     | OHE                            | iterations=926, learning\_rate=0.2963, depth=4, l2\_leaf\_reg=1.8556, bagging\_temperature=0.1028, random\_strength=17.73, border\_count=248  | 0.729    | 0.733    |
| CatBoost (catboost encoding) | OHE                            | iterations=251, learning\_rate=0.1666, depth=8, l2\_leaf\_reg=0.0519, bagging\_temperature=0.0263, random\_strength=3.2946, border\_count=507 | 0.740    | 0.738    |
| GBC (with embeddings)        | OHE + text embeddings          | n\_estimators=303, learning\_rate=0.0540, max\_depth=10, min\_samples\_split=5, min\_samples\_leaf=11, subsample=0.9318, max\_features=`None` | 0.690    | 0.687    |
| CatBoost (with embeddings)   | CatBoost encoding + embeddings | iterations=599, learning\_rate=0.0673, depth=10, l2\_leaf\_reg=0.7998, bagging\_temperature=0.3299                                            | 0.683    | 0.680    |

\---------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- |

---

## 3. Deep Learning

| Model            | Features                         | Hyperparameters                                                                                                                                                                                                    | Accuracy | F1 Score |
| ---------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- |
| Tabular MLP      | Numeric + categorical embeddings | hidden\_layers=\[64, 32, 16], dropout=0.4, embedding\_dims=\[…], epochs=50                                                                                                                                         | 0.5478   | 0.4760   |
| TabNetClassifier | Numeric + categorical embeddings | cat\_emb\_dim=16, optimizer=Adam(lr=2e-2, weight\_decay=1e-5), scheduler=StepLR(step\_size=10, gamma=0.5), mask\_type='entmax', early\_stopping=True, epochs\_trained=35, best\_epoch=25, best\_valid\_auc=0.61065 | 0.5772   | 0.6118   |
| TabTransformer   | Numeric + categorical embeddings | dim=32, depth=6, heads=8, attn\_dropout=0.1, ff\_dropout=0.1, mlp\_hidden\_mults=(4,2), mlp\_act=ReLU, dim\_out=1, continuous\_mean\_std=\[…], epochs=50                                                           | 0.5538   | 0.5363   |

*Notes:*

* **Tabular MLP:** 3 hidden layers (64, 32, 16) with dropout of 0.4, trained for 50 epochs.
* **TabNetClassifier:** embedding dim 16; Adam optimizer (lr=0.02, weight\_decay=1e-5); StepLR scheduler (step\_size=10, gamma=0.5); entmax mask; early stopping at epoch 35 (best at epoch 25 with AUC=0.61065).
* **TabTransformer:** transformer-based model with 32-dimensional embeddings, 6 transformer blocks, 8 heads, dropout rates 0.1; MLP head multipliers of 4 and 2; ReLU activations; run for 50 epochs; final validation Acc/F1 reported at epoch 50.

---
