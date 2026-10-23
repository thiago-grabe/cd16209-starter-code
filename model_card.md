# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Developer and context: trained by Thiago M. Grabe on October 22, 2026 for a Udacity project (starter repository `cd16209-starter-code`). It is a course exercise, not a production model.
- Model type: scikit-learn `HistGradientBoostingClassifier`, a histogram-based gradient boosted decision tree ensemble trained with binary log loss.
- Task: binary classification of the `salary` label. Positive class `>50K`, negative class `<=50K`.
- Hyperparameters, as recorded in `model/metadata.json`. Everything not listed is the scikit-learn 1.7.2 default, and no hyperparameter search was run.

| Hyperparameter | Value |
|---|---|
| max_iter | 200 |
| learning_rate | 0.1 |
| max_leaf_nodes | 31 |
| min_samples_leaf | 20 |
| early_stopping | False |
| random_state | 42 |
| class_weight | None (no reweighting) |
| l2_regularization | 0.0 |
| max_depth | None |
| max_bins | 255 |

- Inputs: 14 Census features.
  - 6 numeric, passed through unchanged: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
  - 8 categorical, one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
  - After encoding the model sees 108 columns. The label is binarized with `LabelBinarizer` (`<=50K` -> 0, `>50K` -> 1).
- Output: a class label and, through the API, the predicted probability of `>50K`. The label follows scikit-learn's default rule: `>50K` when that probability is at least 0.5.
- Software: scikit-learn 1.7.2, numpy 2.3.3, pandas 2.3.2 on Python 3.13.8, macOS arm64.
- Artifacts written by `python -m starter.train_model`:
  - `model/model.pkl`: the fitted classifier.
  - `model/encoder.pkl`: the fitted one-hot encoder.
  - `model/lb.pkl`: the fitted label binarizer.
  - `model/metadata.json`: hyperparameters, feature lists, split sizes, metrics and library versions.
  - `slice_output.txt`: per-slice metrics for every categorical feature.
- Code: `starter/ml/data.py` (loading, cleaning, encoding), `starter/ml/model.py` (training, inference, metrics, persistence), `starter/ml/slices.py` (slice metrics), `starter/train_model.py` (pipeline) and `main.py` (FastAPI service).

## Intended Use

- Primary use: an educational demonstration of a complete tabular machine learning workflow: cleaning a public dataset, training a classifier, evaluating it overall and on data slices, and serving it behind a FastAPI service (`POST /predict`, `POST /predict/batch`) with a small web UI.
- What it predicts: whether a person's annual income exceeds 50K US dollars, from the 14 attributes recorded in the 1994 US Census extract listed above.
- Primary users: learners working through the project and reviewers checking that the pipeline, the tests and the API behave as documented. The API is meant to be exercised with the bundled examples or with made-up records.
- Out of scope: any real decision about a person. In particular the model must not be used for
  - hiring, promotion or compensation decisions,
  - lending, credit scoring or insurance pricing,
  - housing or tenant screening,
  - benefits eligibility, advertising targeting or any other consequential decision.
- Also out of scope: estimating current incomes (the data is from 1994 and the 50K threshold is not inflation adjusted), populations outside the United States, and any causal claim about what drives income. Probabilities from this model describe correlations in a survey that is more than thirty years old, nothing more.

## Training Data

- Source: the bundled `data/census.csv`, the "Adult" extract of the 1994 US Census Bureau Current Population Survey distributed by the UCI Machine Learning Repository (Kohavi and Becker). The file has 32,561 rows, 14 features (6 numeric, 8 categorical) and the label `salary`.
- Label balance: 24,720 rows `<=50K` (75.9%) and 7,841 rows `>50K` (24.1%).
- Cleaning (`starter/ml/data.py`) is deliberately minimal:
  - column names and text values are stripped of incidental whitespace (the raw file has a space after every comma);
  - the `?` placeholder for a missing `workclass` (1,836 rows), `occupation` (1,843) and `native-country` (583) is kept as a category of its own;
  - the 24 exact duplicate rows are kept;
  - there is no imputation, outlier removal or scaling (tree ensembles do not need scaled inputs).
- Split: stratified by `salary` with `train_test_split(test_size=0.2, random_state=42)`, giving 26,048 training rows (19,775 `<=50K`, 6,273 `>50K`) and 6,513 evaluation rows. Stratification keeps the 76/24 class ratio in both splits.
- Encoding: the one-hot encoder for the 8 categorical features and the label binarizer are fitted on the 26,048 training rows only. Numeric features are used as is. The resulting 108-column matrix is what the classifier is trained on.

## Evaluation Data

- The 6,513 held-out rows (20% of the file) from the stratified split above: 4,945 labelled `<=50K` and 1,568 labelled `>50K`. These rows were not used to fit the encoder, the label binarizer or the model.
- The evaluation rows are transformed with the encoder and label binarizer fitted on the training split, exactly as the API transforms incoming requests (`starter.ml.model.predict_labels` loads the same pickled objects). A category absent from the training split is encoded as all zeros (`handle_unknown="ignore"`).
- The same 6,513 rows and the same predictions feed the per-slice report in `slice_output.txt`, so the overall numbers and the slice numbers are directly comparable.

## Metrics

The metrics are precision, recall and F1 (F-beta with beta=1) for the positive class `>50K`, computed with scikit-learn on the 6,513 evaluation rows at the default 0.5 probability threshold. All three are computed with `zero_division=1`, so a slice where a denominator is zero reports 1.0 instead of raising an error (see the slice notes below). Values are rounded to four decimals and copied from `model/metadata.json`.

| Metric | Value |
|--------|-------|
| Precision | 0.7855 |
| Recall | 0.6773 |
| F1 | 0.7274 |

Reading the table: of the records predicted `>50K`, 78.6% really earn more than 50K; of the records that really earn more than 50K, 67.7% are found. The model is more conservative than it is sensitive, which is expected from the 76/24 class imbalance and the unweighted log loss.

### Slice performance

`slice_output.txt` holds precision, recall and F1 for every value of all 8 categorical features (101 slices in total) on the same evaluation rows. The table below lists, for five features, the best and worst F1 among values with at least 100 evaluation rows, so that the ranking is not driven by slices of a handful of records. Every number is copied exactly from `slice_output.txt`.

| Feature | Rank | Value | n | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| sex | best | Male | 4355 | 0.7846 | 0.6856 | 0.7317 |
| sex | worst | Female | 2158 | 0.7908 | 0.6327 | 0.7029 |
| race | best | White | 5533 | 0.7899 | 0.6851 | 0.7338 |
| race | worst | Asian-Pac-Islander | 200 | 0.6809 | 0.6154 | 0.6465 |
| education | best | Prof-school | 114 | 0.8526 | 0.9759 | 0.9101 |
| education | worst | 7th-8th | 120 | 1.0000 | 0.0000 | 0.0000 |
| workclass | best | Self-emp-inc | 222 | 0.7652 | 0.8559 | 0.8080 |
| workclass | worst | ? | 363 | 0.7931 | 0.6389 | 0.7077 |
| native-country | best | ? | 125 | 0.7179 | 0.8750 | 0.7887 |
| native-country | worst | Mexico | 111 | 1.0000 | 0.3333 | 0.5000 |

Notes on the slices:

- Education is the strongest driver of performance. Slices with graduate or professional degrees are easy because most of their members earn more than 50K (`Prof-school` F1 0.9101, `Masters` 0.9037, `Doctorate` 0.8682), while the two largest slices are weak: `HS-grad` (n=2120) has F1 0.5098 with recall 0.3835 and `Some-college` (n=1475) has F1 0.6090.
- On `education=7th-8th` (n=120) the model predicted no positives at all, so recall is 0.0000 and the precision of 1.0000 is the `zero_division=1` convention, not a real score. `native-country=Mexico` shows the same pattern in a milder form: a precision of 1.0000 on only a third of the actual high earners (recall 0.3333) means the model almost never predicts `>50K` for this group.
- The `?` (missing) slices perform about as well as the overall model for `workclass` and better than it for `native-country`; missingness itself is not hurting the model.
- 51 of the 101 slices have fewer than 100 evaluation rows and 44 have fewer than 30. Their values swing between 0.0 and 1.0 and should not be interpreted; see the caveats below.

## Ethical Considerations

- Protected attributes are model inputs. `sex`, `race` and `native-country` are used directly as features. `marital-status`, `relationship` (`Husband` and `Wife` encode sex and marital status together), `occupation` and `fnlgt` (the Census sampling weight, computed from age, race, sex and other demographic controls) act as proxies for them. The model therefore learns and reproduces the demographic income gaps present in the 1994 data; nothing in the pipeline prevents that.
- Performance differs across groups on the held-out split. The rows below are copied from `slice_output.txt`:

| Group | n | Precision | Recall | F1 |
|---|---|---|---|---|
| sex = Male | 4355 | 0.7846 | 0.6856 | 0.7317 |
| sex = Female | 2158 | 0.7908 | 0.6327 | 0.7029 |
| race = White | 5533 | 0.7899 | 0.6851 | 0.7338 |
| race = Black | 662 | 0.7778 | 0.6022 | 0.6788 |
| race = Asian-Pac-Islander | 200 | 0.6809 | 0.6154 | 0.6465 |
| race = Amer-Indian-Eskimo | 73 | 0.6667 | 0.4444 | 0.5333 |
| native-country = United-States | 5835 | 0.7896 | 0.6768 | 0.7289 |
| native-country = Mexico | 111 | 1.0000 | 0.3333 | 0.5000 |

- By sex, precision is about equal (0.7908 for women versus 0.7846 for men) but recall is not: high-earning women are missed more often (0.6327 versus 0.6856), a gap of 0.029 in F1.
- By race, F1 drops from 0.7338 for `White` to 0.6788 for `Black` (recall 0.6022) and 0.6465 for `Asian-Pac-Islander`; `Amer-Indian-Eskimo` is lower still but rests on 73 rows and is not stable.
- By native country, `Mexico` has recall 0.3333, so two out of three high earners born in Mexico are predicted `<=50K`. All other countries have fewer than 50 evaluation rows and their numbers are not meaningful.
- The errors are not symmetric across groups. For women, Black and Asian-Pac-Islander respondents and Mexican-born respondents the dominant error is a false negative, a high earner predicted as `<=50K`. In any screening use this would systematically under-rate those groups, which is one more reason the model must not be used for decisions about people.
- The data is from 1994. Income levels, the labour market, occupational categories and the demographic composition of the United States have changed since then, and the Census category labels (the `race` values, the single binary `sex`) do not reflect how these attributes are collected or self-described today.
- Representation is uneven: 67% of the rows are `Male`, 85% are `White` and 90% have `native-country=United-States`. Minority groups have both less training signal and far fewer evaluation rows, so their reported metrics are less reliable than the overall numbers.
- No fairness mitigation was applied: no reweighting, no removal of protected attributes, no per-group thresholds, no post-processing, and no formal fairness metric (demographic parity, equalized odds or similar) was computed. The slice report measures performance per group; it does not correct anything.
- The dataset is public and anonymised, so the training data itself raises no privacy concern, but the API accepts arbitrary records and nobody should submit real personal data to a demonstration service.

## Caveats and Recommendations

- Class imbalance: 76% of the rows are `<=50K` and 24% are `>50K`. The model is trained with plain log loss and no class weights, so it leans towards the majority class; this is why recall (0.6773) trails precision (0.7855). If missing high earners is the costlier error, retrain with `class_weight="balanced"` or lower the threshold, then re-evaluate.
- Fixed decision threshold: labels use scikit-learn's default rule, `>50K` when the predicted probability is at least 0.5. The API also returns that probability (`probability_over_50k`) so callers can apply a different cut-off, but the metrics in this card only hold at 0.5.
- Unknown categories: the encoder uses `handle_unknown="ignore"`, so a category value that was not in the training split (a typo, a new country name, a different spelling) is encoded as all zeros instead of raising an error. The prediction then relies on the remaining features only, and the service does not flag it. Validate inputs upstream if that matters.
- Small slices are unreliable: 51 of the 101 slices have fewer than 100 evaluation rows, 44 have fewer than 30 and 23 have fewer than 10. For those, a single record moves the metrics by many points, and the `zero_division=1` convention produces
  - precision 1.0000 when the model predicts no positives, for example `native-country=Nicaragua` (n=11, recall 0.0000), and
  - recall 1.0000 when the slice contains no positives, for example `native-country=Jamaica` (n=21, precision 0.0000).
  Always read a slice together with its `n`.
- Single split, no tuning: the hyperparameters were set once and not searched, and no cross-validation was run. With 1,568 positive evaluation rows the overall metrics carry sampling noise of roughly one hundredth, so differences of that size between slices or between retrainings are not meaningful.
- Reproducibility: the split and the model are seeded with `random_state=42`, so retraining on the same platform with the same library versions reproduces these numbers. Retraining on another operating system, CPU architecture or scikit-learn version can shift the last decimals of the metrics and of the slice table; `load_artifacts` warns when the installed scikit-learn differs from the version recorded in `model/metadata.json`.
- Keep this card in sync: `python -m starter.train_model` regenerates `model/` (including `model/metadata.json`) and `slice_output.txt`. After any retraining, copy the new metric values and slice rows into this card and update the training date. `tests/test_docs.py` checks that the metrics table agrees with `model/metadata.json`.
- Pickled artifacts: the `model/*.pkl` files are Python pickles. Only load them from a copy of this repository you trust; never load pickles from an untrusted source.
- If the project is extended, the first improvements to make are: add a per-group fairness metric to the slice report, measure how much performance changes when `sex`, `race` and `native-country` are removed, tune the threshold on a validation split, and switch to a current income dataset for anything beyond the exercise.
