# DataVine Analytics: Machine Learning Case Studies

This project applies a consistent data science workflow to three business scenarios: wine classification, feed similarity recommendations, and regional crime pattern analysis. The work is documented in the Jupyter notebook `DataVine (1).ipynb`.

## Project overview

### 1. Wine classification

A wine dataset with 178 records and 13 chemical measurements is used to predict one of three wine classes. The notebook explores the data, standardizes features, applies PCA while retaining at least 95% of the variance, and tunes k-Nearest Neighbors (k-NN) with 5-fold `GridSearchCV`. It also compares tuned Support Vector Machine (SVM) and Random Forest models.

Saved notebook results show a 38.9% majority-class baseline, 96.4% variance retained by 10 PCA components, and 97.2% test accuracy for k-NN. SVM and Random Forest each reached 100% test accuracy on the notebook's 36-row test split.

### 2. Agricultural feed recommendations

The Chickwts dataset contains 71 chicken weight observations across six feed types. The notebook compares weight distributions and average weights, standardizes weight, applies one-component PCA, then compares feed profiles using cosine similarity. The saved results identify Casein and Sunflower as similar high-weight profiles, and Meatmeal and Linseed as alternatives near Soybean, depending on the profile representation used.

These recommendations are based only on observed weights. They do not account for nutrition, cost, animal health, or other factors needed to make a real feeding decision.

### 3. Regional crime pattern analysis

The USArrests dataset covers 50 U.S. states. The notebook standardizes Murder, Assault, and Rape rates, reduces them to two PCA components, and compares K-Means clustering with a Gaussian Mixture Model (GMM). The two components explain about 93.9% of the variance. K-Means is presented with four clusters; BIC selects two GMM components. The notebook includes cluster visualizations, sizes, crime-rate summaries, and GMM membership probabilities.

## Repository layout

Place the notebook and CSV files in this layout (folder capitalization can be adjusted for your operating system):

```text
project/
├── DataVine (1).ipynb
└── DataSets/
    ├── wine.csv
    ├── Chickwts.csv
    └── USArrests.csv
```

The notebook reads the datasets from relative paths. In the supplied notebook, the wine path is written as `dataSets/wine.csv`, while the other two paths use `DataSets/`. These spellings resolve to the same folder on Windows, but on case-sensitive systems update the wine path to `DataSets/wine.csv` (or rename the directory consistently).

The CSV files are not included with this README. Obtain the datasets from the course materials and ensure the column names match the notebook's expectations:

- `wine.csv`: numeric feature columns and a `target` column.
- `Chickwts.csv`: `weight` and `feed` columns.
- `USArrests.csv`: `rownames`, `Murder`, `Assault`, `UrbanPop`, and `Rape` columns.

## Requirements

Use Python 3 with Jupyter Notebook or JupyterLab. The notebook imports:

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install the packages in your Python environment if they are not already available:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run the notebook

1. Download or copy the three CSV files into the `DataSets/` folder using the layout above.
2. Open a terminal in the project directory and start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `DataVine (1).ipynb` and run the cells from top to bottom.

Grid search and plots may take a short while to complete. The notebook has saved outputs for the reported results, but rerunning cells will recompute them from the supplied datasets.

## Notes

- Wine results are from a single `train_test_split` call (without stratification) with `test_size=0.2` and `random_state=42`; the 36-row test set is small, so perfect test accuracy should not be taken as a guarantee of performance on new data.
- The feed notebook computes cosine similarity both from one PCA value per feed and from standardized mean-and-standard-deviation profiles. Because the PCA representation is one-dimensional, cosine scores can be `-1` or `1`; the profile-based comparison gives more nuanced rankings.
- Crime clusters are descriptive groupings in this dataset, not causal findings or policy recommendations. Cluster numbers are arbitrary labels and can change between runs or implementations.

