House Prices: manual preprocessing, imputation experiment, CV
Practice project on the Kaggle House Prices: Advanced Regression Techniques data (Ames, Iowa; ~1,460 training houses, ~79 features). The goal was to build the full workflow from scratch, understand each step, and test one change with proper validation.
Notebook: house_prices_baseline_xgboost.ipynb
Approach
Split first, preprocess after. 75/25 train/validation split (random_state=0). All preprocessing is fit on the training set only, to avoid leakage.
Manual preprocessing (no Pipeline).
Numeric columns: median imputation
Categorical columns: most-frequent imputation, then one-hot encoding (handle_unknown='ignore')
Index and column names restored by hand so the pieces line up before concatenation
Model: RandomForestRegressor, scored with MAE on the validation set.
Rebuilt with ColumnTransformer + Pipeline and compared using 5-fold cross-validation.
Results
Setup	MAE
Baseline, manual preprocessing, single split	$17,174
Categorical blanks filled with 'None', single split	$17,053
5-fold CV, most-frequent fill	$17,672
5-fold CV, 'None' fill	$17,691
The experiment
In many categorical columns (Alley, PoolQC, Fence, ...) a blank means "this house doesn't have one," not "unknown." The baseline filled those blanks with the most common value, so the model was told nearly every house had an alley. I replaced that with the string 'None'.
On a single split MAE dropped by ~$120 (0.7%). Under 5-fold CV the difference disappeared (~$18, slightly worse). Conclusion: the change doesn't matter for a random forest on this data, and the single-split improvement was noise. Small differences on one validation split of 365 houses can't be trusted, so CV is the number to believe.
What I learned
Fit imputers and encoders on training data only; transform validation and test data.
train_test_split returns pieces by position, so the unpacking order matters.
Restoring the index after sklearn transformers is necessary for pd.concat to align rows correctly.
A single-split win is not evidence; cross-validation is.
Next
Add XGBRegressor and compare it to the random forest using the same 5-fold CV.
Feature engineering (total square footage, house age).
Fit on log(SalePrice) to match the competition's metric.
Environment
Python 3.11, pandas, numpy, scikit-learn. Run on Kaggle notebooks.
