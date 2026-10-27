# Classical Machine Learning
In this part we will cover the classic machine learning algorithms.
In particular, we will discuss the following algorithms:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines

On top of these, we cover gradient-boosted decision trees (the workhorse for tabular data), how to explain
the predictions of a model, and how to track, register, serve and monitor models (MLOps).
"Classical" does not mean outdated: for structured/tabular data, a well-tuned gradient-boosted tree model or even
a regularised linear model is very often the best choice, also in the age of deep learning. These models are fast to
train on a CPU, need comparatively little data, and are much easier to explain and to operate.

The part is organised in four chapters:

- **c00 Linear models (and friends):** ordinary least squares from scratch (normal equation and gradient descent),
  regularisation (Ridge, Lasso, ElasticNet), residual diagnostics, logistic regression with proper evaluation
  (ROC-AUC, precision/recall, calibration, threshold selection) and a tour of decision trees, random forests and SVMs.
- **c01 Gradient-boosted trees:** boosting from scratch, then scikit-learn's HistGradientBoosting, XGBoost and LightGBM
  with early stopping and hyperparameter search (random search and Optuna).
- **c02 Model explainability:** SHAP values, permutation importance, partial dependence and ICE curves - and the
  pitfalls of feature importance.
- **c03 MLOps:** experiment tracking, model registry, serving and drift monitoring with MLflow.

Most examples use small datasets that ship with scikit-learn (diabetes, breast cancer) or a synthetic customer churn
dataset (`c01_gradient_boosted_trees/utils/churn_data.py`). Synthetic data has a big didactic advantage: we know the
true data-generating process, so we can check whether a model - and its explanation - recovers the real effects.
All scripts run offline on a CPU in well under two minutes each.

## The machine learning workflow
Every chapter follows the same basic workflow. It is worth spelling it out once, because most mistakes in applied
machine learning are workflow mistakes rather than algorithmic ones.

1. **Define the problem and the metric.** What is predicted (regression: a number; classification: a class or a
   probability)? Which errors are expensive? The metric must reflect the business goal - e.g. recall of malignant
   tumours, or the log-loss of churn probabilities that are fed into a campaign budget.
2. **Split the data before looking too closely at it.**
   - *Training set*: used to fit the model parameters.
   - *Validation set*: used to choose hyperparameters, the number of boosting rounds (early stopping), the decision
     threshold, or between model families.
   - *Test set*: used **once**, at the very end, to estimate the performance on new data. Every decision made by
     looking at the test set leaks information and makes the estimate optimistic.

   Typical splits are 60/20/20 or 80/20 with cross-validation on the 80%. Classification splits should be
   *stratified* (same class proportions in every split); time-series data must be split by time, and grouped data
   (several rows per customer/patient) by group.
3. **Cross-validation.** With little data, a single validation split is noisy. In $K$-fold cross-validation the
   training data are split into $K$ folds; the model is trained $K$ times on $K-1$ folds and evaluated on the remaining
   one. The mean (and standard deviation) of the $K$ scores is a much more stable estimate, used e.g. by
   `GridSearchCV`/`RandomizedSearchCV`. Out-of-fold predictions (`cross_val_predict`) give a validation prediction
   for every training sample, which is useful e.g. for threshold selection.
4. **Preprocess inside a pipeline.** Scaling, encoding and imputation are *fitted* on the training data only.
   scikit-learn's `Pipeline` and `ColumnTransformer` guarantee this also inside cross-validation, preventing
   *data leakage*.
5. **Start with a baseline**, e.g. a (regularised) linear model or a random forest with default settings. A complex
   model that does not clearly beat a simple baseline is not worth its operational cost.
6. **Evaluate with the right metrics.**
   - Regression: RMSE $=\sqrt{\frac{1}{n}\sum_i (y_i-\hat y_i)^2}$ (same unit as the target), MAE (robust to
     outliers), $R^2 = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}$ (fraction of variance explained).
   - Classification: the confusion matrix and the metrics derived from it - accuracy, precision
     $=\frac{TP}{TP+FP}$, recall $=\frac{TP}{TP+FN}$ and F1 (their harmonic mean) - all depend on a decision
     threshold. Threshold-free metrics evaluate the ranking (ROC-AUC, average precision) or the probabilities
     themselves (log-loss $-\frac{1}{n}\sum_i [y_i\log p_i + (1-y_i)\log(1-p_i)]$, Brier score
     $\frac{1}{n}\sum_i (p_i - y_i)^2$). With imbalanced classes, accuracy is misleading and precision/recall or
     average precision are more informative.
7. **Understand the model** (chapter c02) and check that it behaves sensibly.
8. **Deploy, monitor and retrain** (chapter c03).

### Bias, variance and regularisation
The expected squared error of a model on new data decomposes into
$\text{bias}^2 + \text{variance} + \text{irreducible noise}$. Simple models (a linear model on a non-linear problem,
a very shallow tree) have high bias: they underfit. Very flexible models (a fully grown tree, an RBF-SVM with a huge
$\gamma$) have high variance: they fit the noise of the particular training set and overfit. The gap between
training and validation performance is the practical symptom. *Regularisation* - penalising large coefficients,
limiting tree depth, shrinkage in boosting, averaging many trees - trades a little bias for a large reduction in
variance. The amount of regularisation is a hyperparameter and is chosen on validation data, which is exactly why
we need the train/validation/test discipline above.
