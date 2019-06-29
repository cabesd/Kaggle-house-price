import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns



def summary_missing_data(df: pd.DataFrame, lowest_proportion: float = 0.0) -> pd.DataFrame:
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Count', 'Percent'])
    return missing_data[missing_data['Percent'] > lowest_proportion]

def pearson_correlation_heatmap(df: pd.DataFrame, size=25) -> None:
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=size)
    
def _eval_components(Z):
    for i in range(1, Z.shape[1], 1):
        pca = PCA(n_components=i).fit(Z)
        # print('Variance ratio = ', pca.explained_variance_ratio_)
        print('Cumulative sum:\t', sum(pca.explained_variance_ratio_), '\twith', pca.n_components_, 'components')
        
def evaluate_regressor(model, X, Y, name=None, nruns=200, other_metric=None):
    r2, mse, extra = [], [], []
    for j in range(nruns):
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
        model.fit(xtrain, ytrain)
        YP = model.predict(xtest)
        r2.append(r2_score(YP, ytest))
        mse.append(mean_squared_error(YP, ytest))
        if other_metric!=None:
            keep_positives = YP >= 0
            extra.append(other_metric['call'](YP[keep_positives], ytest[keep_positives]))
    print("Runs:\t\t", nruns)
    print("Mean R2:\t", np.mean(r2), "\nSTD R2:\t\t", np.std(r2))
    print("Mean MSE:\t", np.mean(mse), "\nSTD MSE:\t", np.std(mse))
    if other_metric!=None: print(other_metric['name']+":\t\t", np.mean(extra))
    plt.hist(r2)
    plt.title("R2 Histogram - "+name)
    plt.xlim(0, 1)
    
def show_grid_results(grid_search, all=True):
    print('Best parameters:\n', grid_search.best_params_, '\n', grid_search.best_score_, '\n')
    if all:
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(mean_score, params)