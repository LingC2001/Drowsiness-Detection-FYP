import pandas as pd
from mrmr import mrmr_regression
from sklearn.feature_selection import SelectKBest, f_regression

if __name__ =="__main__":
    # Loading Dataset as pandas
    folder = "D:/School/Drowsiness-Detection-FYP/neural_network/Features_and_Labels/30s/"
    t = pd.read_csv(folder + "train_features_and_labels_30s.csv")
    v = pd.read_csv(folder + "valid_features_and_labels_30s.csv")
    
    dataset = pd.concat([t, v], axis=0)

    X = dataset.iloc[:,0:-4]
    y = dataset.iloc[:,-1]
    print(X)
    print(y)
    # MRMR selection
    selected_features_MRMR = mrmr_regression(X=X,y=y, K=60)
    print(selected_features_MRMR)

    # SelecKBest selection
    selector = SelectKBest(f_regression, k=60)
    selector.fit(X, y)
    cols_idxs = selector.get_support(indices=True)
    features_df_new = X.iloc[:,cols_idxs]
    print(features_df_new.columns.to_list())
