import pandas as pd
from mrmr import mrmr_regression, mrmr_classif
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

if __name__ =="__main__":
    # Loading Dataset as pandas
    folder = "D:/School/Drowsiness-Detection-FYP/neural_network/Features_and_Labels/30s/"
    t = pd.read_csv(folder + "train_features_and_labels_30s.csv")
    v = pd.read_csv(folder + "valid_features_and_labels_30s.csv")
    
    dataset = pd.concat([t, v], axis=0)

    X = dataset.iloc[:,0:-4]
    y = dataset.iloc[:,-1]

    y = y.to_numpy()
    y[y < 6.5] = 0
    y[y >= 6.5] = 1
 
    print(X)
    print(y)


    # MRMR selection
    selected_features_MRMR = mrmr_classif(X=X,y=y, K=30)
    print(selected_features_MRMR)

    # SelecKBest selection
    selector = SelectKBest(f_classif, k=30)
    selector.fit(X, y)
    cols_idxs = selector.get_support(indices=True)
    features_df_new = X.iloc[:,cols_idxs]
    print(features_df_new.columns.to_list())
