## Gradient-Boosted Trees
Gradient-boosted decision trees (GBTs) are the default choice for supervised learning on tabular data. They win most
tabular competitions, handle mixed feature types, missing values, non-linear effects and interactions without
feature engineering, and train in seconds on a CPU. XGBoost, LightGBM, CatBoost and scikit-learn's
`HistGradientBoosting*` are all implementations of the same idea.

This chapter uses a synthetic customer churn dataset (`utils/churn_data.py`): 20,000 customers with tenure, monthly
and total charges, contract type, payment method, fibre internet, number of support calls, age, number of products
and a pure-noise column; about 30% churn. The true churn log-odds contain a non-linear tenure effect, a U-shaped age
effect, a saturating effect of support calls and two interactions (price $\times$ contract type, fibre $\times$
support calls). Because we know the true probabilities, the scripts also print the **best achievable** score - a
ceiling that is never available on real data.

Scripts:

- `main_1_boosting_from_scratch.py`: gradient boosting for regression and (Newton-)boosting for classification in a
  few lines of numpy, verified against scikit-learn; learning rate vs number of trees.
- `main_2_gbt_comparison.py`: HistGradientBoosting vs XGBoost vs LightGBM with early stopping, compared with a
  logistic regression and a random forest baseline.
- `main_3_hyperparameter_search.py`: `RandomizedSearchCV` and an optional Optuna study.

### Boosting as an additive model
Random forests average many *independent* deep trees to reduce variance. Boosting instead builds an **additive
model** of many *small* trees sequentially, each one correcting the errors of the ensemble so far:
$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta\, h_m(x),$$
where $F_0$ is a constant (the mean target, or the log-odds of the base rate), $h_m$ are shallow regression trees
and $\eta\in(0,1]$ is the **learning rate** (shrinkage). Each tree reduces the bias of the ensemble a little.

**Gradient boosting** (Friedman, 2001) chooses $h_m$ by functional gradient descent on a differentiable loss
$L(y, F)$. At iteration $m$ we compute for every training sample the negative gradient (the "pseudo-residual")
$$r_{im} = -\left.\frac{\partial L(y_i, F)}{\partial F}\right|_{F = F_{m-1}(x_i)},$$
fit a regression tree $h_m$ to the pairs $(x_i, r_{im})$, set the value of each leaf optimally for the loss, and add
the tree with step size $\eta$. For the squared loss $L=\frac{1}{2}(y-F)^2$ the negative gradient is simply the
residual $y_i - F_{m-1}(x_i)$: *each tree is fitted to the residuals of the current ensemble*.
`main_1_boosting_from_scratch.py` implements exactly this in ~15 lines and reproduces scikit-learn's
`GradientBoostingRegressor` to machine precision; `outputs/boosting_stages_regression.png` shows the fit after 1, 5,
20 and 200 trees of depth 2.

For binary classification we work on the log-odds scale, $p = \sigma(F)$, with the log-loss. Its gradient and
Hessian with respect to $F$ are
$$g_i = p_i - y_i, \qquad h_i = p_i(1 - p_i).$$
A tree fitted to $-g_i$ determines the *structure* (the splits); the value of leaf $j$ is then set by one Newton
step on the samples $I_j$ in that leaf,
$$w_j = \frac{\sum_{i\in I_j} (y_i - p_i)}{\sum_{i\in I_j} p_i(1-p_i) + \lambda},$$
with an optional L2 penalty $\lambda$ on the leaf values. This **second-order** view is the core of XGBoost.

### What XGBoost and LightGBM add
XGBoost (Chen & Guestrin, 2016) minimises at every iteration the second-order Taylor approximation of a
*regularised* objective
$$\mathcal{L}^{(m)} \approx \sum_i \left[g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2\right] + \gamma T + \tfrac{1}{2}\lambda\sum_{j=1}^T w_j^2,$$
where $T$ is the number of leaves. Plugging in the optimal leaf weights $w_j^* = -G_j/(H_j+\lambda)$ (with
$G_j=\sum_{i\in I_j} g_i$, $H_j=\sum_{i\in I_j} h_i$) gives a closed-form **gain** for every candidate split, which is
used to grow the tree directly on the second-order statistics; $\gamma$ is the minimum gain needed to make a split.
On top of this come engineering advances: histogram-based split finding (features are bucketed into ~256 bins, so
split search costs $O(\text{bins})$ instead of $O(n)$), sparsity-aware handling of missing values (each split learns a
default direction), row and column subsampling, and parallel/out-of-core/GPU training.

LightGBM (Ke et al., 2017) uses the same histogram approach but grows trees **leaf-wise** (always splitting the leaf
with the largest gain, controlled by `num_leaves`) instead of level-wise, which reaches a lower loss with fewer
leaves but can overfit small data (use `min_child_samples`). It adds Gradient-based One-Side Sampling (GOSS: keep the
samples with large gradients, subsample the rest) and Exclusive Feature Bundling (EFB: merge sparse, mutually
exclusive features). scikit-learn's `HistGradientBoostingClassifier` is a LightGBM-inspired implementation with
native support for missing values and categorical features and no extra dependency. CatBoost (not used here)
specialises in categorical features with ordered target statistics.

All three libraries handle **categorical features natively** if they are passed as pandas `category` columns
(`categorical_features="from_dtype"` in scikit-learn, `enable_categorical=True` in XGBoost, automatic in LightGBM):
splits partition the categories into two groups, which avoids huge one-hot encodings.

### The important hyperparameters
- **Number of trees and learning rate.** They are coupled: a smaller learning rate needs proportionally more trees
  but generally generalises better (`outputs/boosting_learning_rate.png`). The training loss keeps decreasing
  forever, while the validation loss is U-shaped. In practice one fixes a small learning rate (0.01-0.1), allows a
  large number of trees, and lets **early stopping** pick the number: training stops when the validation loss has not
  improved for, say, 50 rounds.
- **Tree complexity:** `max_depth` (XGBoost), `num_leaves` (LightGBM), `max_leaf_nodes` (scikit-learn), and the
  minimum number of samples / Hessian sum per leaf. Depth also bounds the order of interactions a single tree can
  model.
- **Randomisation:** row subsampling (`subsample`) and feature subsampling (`colsample_bytree`) per tree reduce
  variance - *stochastic gradient boosting*.
- **Regularisation:** L2 (`reg_lambda`) / L1 (`reg_alpha`) penalties on leaf values, minimum split gain.

### Comparing the libraries fairly
`main_2_gbt_comparison.py` uses one stratified 60/20/20 train/validation/test split for all models. The baselines are
trained on the training set; the boosters additionally use the validation set for early stopping (learning rate
0.05, up to 2000 trees, patience 50). The test set is used once at the end. A typical result:

| model | test ROC-AUC | test log-loss | trees | fit time |
|---|---|---|---|---|
| logistic regression (one-hot) | 0.836 | 0.445 | - | 0.1 s |
| random forest | 0.847 | 0.436 | 300 | 3 s |
| HistGradientBoosting | 0.849 | 0.430 | 176 | 0.5 s |
| XGBoost | 0.850 | 0.429 | 159 | 0.4 s |
| LightGBM | 0.850 | 0.430 | 117 | 0.3 s |
| *true probabilities* | *0.854* | *0.423* | | |

Observations:

- All three boosting libraries are essentially tied - library choice matters far less than a correct validation
  setup. They close most of the gap between the linear baseline and the best achievable score, because they
  discover the non-linearities and interactions on their own.
- The logistic regression is a strong baseline; it would get closer with hand-crafted features (log-tenure, age
  squared, interaction terms) - which is exactly the feature engineering that trees make unnecessary.
- `outputs/gbt_learning_curves.png` shows the validation log-loss per round and the early-stopping point. Note that
  XGBoost and LightGBM keep the best iteration, while scikit-learn's HGB keeps all trees up to the point where it
  stopped (best + patience).
- **Threads:** for a dataset of this size, a single thread per model is fastest. Multithreaded OpenMP code can even
  become dramatically slower when the machine is busy (oversubscription), which is why the scripts set `n_jobs=1`
  and limit OpenMP threads with `threadpoolctl`. For large datasets use all cores.

The gain-based feature importances of XGBoost and LightGBM (`outputs/gbt_feature_importance.png`) rank the
contract type first and give small but non-zero importance to `total_charges` and `noise`, which have *no* true
effect. Gain importance is computed on the training data and rewards features that offer many split points - see the
next chapter for better tools.

### Hyperparameter search
Defaults are a reasonable start, but tuning typically buys a few per mille of log-loss or AUC.
`main_3_hyperparameter_search.py` compares

1. **LightGBM with defaults** (100 trees, 31 leaves, learning rate 0.1);
2. **Random search with cross-validation** (`RandomizedSearchCV`, 20 configurations $\times$ 3 folds, optimising
   the log-loss): hyperparameters are sampled from distributions - log-uniform for learning rate and penalties,
   integers for leaves and minimum leaf sizes. Random search is more efficient than grid search when only a few
   hyperparameters really matter (Bergstra & Bengio, 2012). Configurations and folds run in parallel processes;
3. **Optuna** (optional, `pip install optuna`): the Tree-structured Parzen Estimator (TPE) proposes promising
   configurations based on previous trials, and each trial uses early stopping on the validation set, so the number
   of trees need not be searched. The final model is refitted with the best number of trees.

Typical test results: defaults 0.843 ROC-AUC / 0.437 log-loss, random search 0.849 / 0.430, Optuna 0.850 / 0.428.
Both searches gain mainly from a much lower learning rate (about 0.02 instead of 0.1) combined with more trees and
row/column subsampling. The random-search winner even uses *more* leaves than the default (39 vs 31) but with a large
minimum leaf size (`min_child_samples` 148 instead of 20); Optuna instead chose very small trees (6 leaves). Several
quite different configurations reach almost the same CV log-loss (the top 5 of the random search lie within 0.003),
so the "best" hyperparameters should not be read as a unique optimum. Differences of about 0.001 are within the noise of a 4,000-row test
set; do not over-interpret them, and never tune on the test set.
