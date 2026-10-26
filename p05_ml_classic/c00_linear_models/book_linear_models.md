## Linear Models, Trees and Support Vector Machines
Linear models are the natural starting point of classical machine learning: they are fast, have few
hyperparameters, can be solved exactly, and their coefficients are directly interpretable. Many ideas that recur in
the rest of the book - loss functions, gradient descent, regularisation, probability calibration, the choice of a
decision threshold - are easiest to understand on a linear model. The chapter closes with a short tour of decision
trees, random forests and support vector machines, which handle non-linear relationships.

The chapter contains three scripts:

- `main_1_linear_regression.py`: OLS from scratch vs scikit-learn, Ridge/Lasso/ElasticNet with cross-validated
  regularisation, regularisation paths and residual diagnostics on the diabetes dataset.
- `main_2_logistic_regression.py`: logistic regression on the breast cancer dataset with a scaling pipeline,
  hyperparameter search, ROC/PR curves, calibration and threshold selection.
- `main_3_trees_rf_svm.py`: decision boundaries of trees, random forests and SVMs, cost-complexity pruning,
  out-of-bag error and a cross-validated comparison of all model families.

### Linear regression
Given $n$ samples with $p$ features collected in the matrix $X\in\mathbb{R}^{n\times p}$ (plus a column of ones for
the intercept) and a target vector $y$, linear regression models $\hat y = X\beta$. Ordinary least squares (OLS)
chooses the coefficients that minimise the squared error
$$L(\beta) = \frac{1}{n}\lVert y - X\beta\rVert_2^2 .$$
Setting the gradient $\nabla L = -\frac{2}{n}X^\top(y - X\beta)$ to zero gives the **normal equation**
$X^\top X\beta = X^\top y$, i.e. $\hat\beta = (X^\top X)^{-1}X^\top y$ when $X^\top X$ is invertible. In code we
solve the linear system (`np.linalg.solve`) rather than computing the inverse, which is faster and numerically more
stable; scikit-learn's `LinearRegression` uses a least-squares solver based on a matrix decomposition.

For very large datasets, or for models without a closed-form solution, we use **gradient descent** instead:
starting from $\beta_0 = 0$ we repeatedly step against the gradient, $\beta_{t+1} = \beta_t - \eta\nabla L(\beta_t)$,
with learning rate $\eta$. The loss of linear regression is convex, so gradient descent converges to the same
solution - quickly if the features are standardised (similar scales make the loss surface round rather than a long,
narrow valley). The script shows that the normal equation, gradient descent and scikit-learn agree to about
$10^{-12}$.

Under the classical assumptions (linearity, independent errors with constant variance, no perfect collinearity)
OLS is the best linear unbiased estimator. In practice the assumptions are checked with **residual diagnostics**,
which the script saves to `outputs/linreg_residuals.png`:

- *Residuals vs fitted values* should look like a structureless band around zero. A curve indicates a missing
  non-linearity; a funnel shape indicates heteroscedasticity (variance growing with the prediction), in which case a
  log-transform of the target often helps. On the diabetes data the Spearman correlation between $|$residual$|$ and
  the fitted value is about 0.36 - a mild funnel.
- The *Q-Q plot* compares the residual quantiles with those of a normal distribution. Normality matters for
  classical confidence intervals and p-values, much less for prediction.

### Regularisation: Ridge, Lasso and ElasticNet
When there are many (correlated) features compared to the number of samples, the OLS coefficients get large and
unstable and the model overfits. Regularisation adds a penalty on the size of the coefficients:

- **Ridge (L2):** $\min_\beta \frac{1}{n}\lVert y - X\beta\rVert_2^2 + \alpha\lVert\beta\rVert_2^2$. Closed form
  $\hat\beta = (X^\top X + n\alpha I)^{-1}X^\top y$; coefficients shrink smoothly towards zero but rarely become
  exactly zero. Ridge handles correlated features gracefully by sharing the weight between them.
- **Lasso (L1):** $\min_\beta \frac{1}{2n}\lVert y - X\beta\rVert_2^2 + \alpha\lVert\beta\rVert_1$. The corners of
  the L1 ball produce *sparse* solutions: many coefficients are exactly zero, so the Lasso performs feature selection.
  It is solved by coordinate descent.
- **ElasticNet:** a mix $\alpha\left(\rho\lVert\beta\rVert_1 + \frac{1-\rho}{2}\lVert\beta\rVert_2^2\right)$ with
  `l1_ratio` $\rho$, combining sparsity with the stability of Ridge for groups of correlated features.

The penalty treats all coefficients equally, so the features **must be standardised** first, and the intercept is
not penalised. The regularisation strength $\alpha$ is chosen by cross-validation (`RidgeCV`, `LassoCV`,
`ElasticNetCV`). From a Bayesian point of view, Ridge corresponds to a Gaussian prior on the coefficients and Lasso
to a Laplace prior.

To make regularisation matter, the script expands the 10 diabetes features to all 65 degree-2 polynomial terms
(squares and pairwise products) for only ~330 training samples. Typical results on the test set:

| model | RMSE | $R^2$ |
|---|---|---|
| OLS, 10 raw features | 53.4 | 0.485 |
| OLS, 65 polynomial features | 56.4 | 0.424 |
| RidgeCV, 65 features | 53.2 | 0.488 |
| LassoCV, 65 features (14 non-zero) | 53.0 | 0.493 |

The unregularised polynomial model overfits; the regularised versions recover (and slightly beat) the simple model,
and the Lasso does so with only 14 of 65 terms. The *regularisation paths* in
`outputs/linreg_regularisation_paths.png` show how the coefficients shrink as $\alpha$ grows: smoothly for Ridge,
one after another to exactly zero for the Lasso.

### Logistic regression
For binary classification ($y\in\{0,1\}$) logistic regression models the log-odds as a linear function,
$$\log\frac{p(x)}{1-p(x)} = \beta_0 + x^\top\beta \quad\Leftrightarrow\quad p(x) = \sigma(\beta_0 + x^\top\beta)
= \frac{1}{1+e^{-(\beta_0 + x^\top\beta)}},$$
and is fitted by maximum likelihood, i.e. by minimising the **log-loss** (binary cross-entropy)
$$L(\beta) = -\frac{1}{n}\sum_{i=1}^n \left[y_i\log p_i + (1-y_i)\log(1-p_i)\right] + \text{penalty}.$$
There is no closed-form solution, but the loss is convex, so solvers such as L-BFGS, Newton methods or SAGA find the
global optimum. A coefficient $\beta_j$ is the change in log-odds for a one-unit increase of feature $j$ (on
standardised features: per standard deviation); $e^{\beta_j}$ is the corresponding odds ratio. In scikit-learn the
penalty strength is parameterised by $C = 1/\alpha$ (small $C$ = strong regularisation) and the L1/L2 mix by
`l1_ratio` (the older `penalty` argument is deprecated).

`main_2_logistic_regression.py` predicts whether a breast tumour is malignant from 30 features of a cell-nucleus
image. We relabel the data so that 1 = malignant: missing a cancer (false negative) is the costly error, so it
should be the positive class. The script

1. builds a `Pipeline(StandardScaler, LogisticRegression)` - the scaler is re-fitted inside every cross-validation
   fold, which prevents leakage;
2. tunes `C` and `l1_ratio` with `GridSearchCV` (5-fold stratified CV, ROC-AUC). A typical result is
   $C\approx0.3$ with an ElasticNet penalty, keeping 24 of 30 coefficients, and a test ROC-AUC of about 0.998;
3. reports the classification report, ROC and precision-recall curves (`outputs/logreg_roc_pr.png`).

#### Calibration
A classifier is *calibrated* if, among all cases that receive a predicted probability of 0.8, about 80% are
positive. Calibration matters whenever probabilities are used as such (expected costs, risk scores, combining
models). The **reliability diagram** (`outputs/logreg_calibration.png`) plots the observed fraction of positives
against the mean predicted probability per bin; the **Brier score** summarises calibration and discrimination in one
number. Logistic regression, which directly optimises the log-loss, is usually well calibrated. Gaussian naive Bayes,
in contrast, assumes independent features and pushes probabilities towards 0 and 1 - its Brier score is almost three
times worse here. Wrapping it in `CalibratedClassifierCV` (isotonic or sigmoid/Platt scaling, fitted on
cross-validation folds) repairs much of this. Note that calibration does not change the ranking much: ROC-AUC
measures only the ordering of the scores, not whether the probabilities are right.

#### Choosing the decision threshold
`predict()` uses a threshold of 0.5, which is only optimal if both errors cost the same and the probabilities are
calibrated. In the cancer example the requirement is a recall of at least 98% for malignant tumours. The threshold
must be chosen on data that are *not* the test set: we use out-of-fold predictions on the training set
(`cross_val_predict`) and choose the largest threshold whose recall is still at least 0.98 (alternatively, the
threshold that maximises F1). Typical outcome on the test set (53 malignant, 90 benign):

| threshold rule | threshold | missed cancers (FN) | false alarms (FP) |
|---|---|---|---|
| default | 0.50 | 3 | 0 |
| max F1 | 0.44 | 2 | 1 |
| recall $\geq$ 0.98 | 0.13 | 1 | 7 |

Lowering the threshold trades false alarms for fewer missed cancers - a business decision that the model cannot
make for us. scikit-learn also offers `TunedThresholdClassifierCV`, which automates this cross-validated threshold
search for a given metric.

### Decision trees
A decision tree recursively splits the feature space into rectangles with rules like "worst perimeter $\le$ 112.8".
At every node the split is chosen greedily to maximise the decrease in impurity - Gini impurity
$\sum_k p_k(1-p_k)$ or entropy $-\sum_k p_k\log p_k$ for classification, squared error for regression - and each
leaf predicts the majority class (or mean). Trees need no feature scaling, handle non-linearities and interactions
automatically and are easy to read (`export_text` prints the rules). Their weakness is **high variance**: a fully
grown tree memorises the training data (training accuracy 1.0 on the noisy two-moons data in
`outputs/decision_boundaries.png`) and small changes in the data produce a completely different tree.

Complexity is controlled with `max_depth`, `min_samples_leaf` or **minimal cost-complexity pruning**: a full tree
is grown and then pruned by minimising $R_\alpha(T) = R(T) + \alpha|T|$, the training error plus $\alpha$ times the
number of leaves. `cost_complexity_pruning_path` returns the sequence of effective $\alpha$ values at which subtrees
are pruned; we pick `ccp_alpha` by cross-validation, which shrinks the breast cancer tree from 21 to 7 leaves. On a
single test set of 143 samples the full and the pruned tree are within noise of each other, but the pruned tree
is far easier to read and more stable.

### Random forests
A random forest averages many decorrelated trees: each tree is trained on a bootstrap sample of the rows
(*bagging*), and at each split only a random subset of the features is considered (`max_features`, by default
$\sqrt{p}$ for classification). Averaging $B$ trees with variance $\sigma^2$ and pairwise correlation $\rho$ gives a
variance of $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$: more trees remove the second term, feature subsampling
reduces $\rho$. Random forests therefore keep the low bias of deep trees while strongly reducing their variance,
and they work well with almost no tuning.

Each tree leaves out about 37% of the rows ($(1-1/n)^n\approx e^{-1}$); predicting every sample with only the trees
that did not see it gives the **out-of-bag (OOB) score**, a free validation estimate (`oob_score=True`). The script
grows the forest incrementally with `warm_start=True` and plots the OOB error against the number of trees
(`outputs/rf_oob_error.png`): beyond a few hundred trees nothing changes - more trees never overfit, they only cost
time. The impurity-based `feature_importances_` of a forest are convenient but biased; chapter c02 explains why and
what to use instead.

### Support vector machines
A linear SVM looks for the hyperplane $w^\top x + b = 0$ with the largest *margin* to the closest training points
(the support vectors). With slack for misclassified points this is the soft-margin problem
$$\min_{w,b}\ \frac{1}{2}\lVert w\rVert^2 + C\sum_i \max\left(0,\ 1 - y_i(w^\top x_i + b)\right),\quad y_i\in\{-1,1\},$$
i.e. the **hinge loss** plus an L2 penalty; $C$ trades margin width against training errors. The solution depends
only on inner products $x_i^\top x_j$, which can be replaced by a **kernel** $k(x_i, x_j)$ - an inner product in a
(possibly infinite-dimensional) feature space - without ever computing that space (the *kernel trick*). The RBF
kernel $k(x, x') = \exp(-\gamma\lVert x - x'\rVert^2)$ produces smooth non-linear boundaries; $\gamma$ sets the
radius of influence of each support vector. Large $C$ and $\gamma$ give wiggly, overfitting boundaries (training
accuracy 1.0, test 0.83 on the moons), small values give smooth ones. SVMs require standardised features,
are tuned with a grid search over $C$ and $\gamma$, scale roughly quadratically with the number of samples, and do
not output probabilities by default (`SVC(probability=True)` adds an internal Platt calibration).

### Comparing model families fairly
The last step of `main_3_trees_rf_svm.py` evaluates all model families with **the same** 5-fold stratified splits on
the breast cancer training data:

| model | CV accuracy | CV ROC-AUC |
|---|---|---|
| logistic regression | 0.972 $\pm$ 0.016 | 0.992 $\pm$ 0.008 |
| decision tree (pruned) | 0.937 $\pm$ 0.018 | 0.936 $\pm$ 0.016 |
| random forest | 0.962 $\pm$ 0.019 | 0.988 $\pm$ 0.009 |
| RBF SVM (tuned) | 0.976 $\pm$ 0.011 | 0.995 $\pm$ 0.005 |

On this small, almost linearly separable dataset the linear model and the SVM are best, a single tree is clearly
worst, and the differences between the top models are within one standard deviation - a reminder to report the
spread of CV scores and to prefer the simpler model when scores are tied. On larger tabular datasets with
non-linear effects and interactions, tree ensembles usually take the lead, which brings us to gradient boosting in
the next chapter.
