# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:37:30 2022

@author: cheng164
"""

import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline

from utilities import tree_accuracy_vs_alpha, validation_curve_plot, GridSearchCV_result, learning_curve_plot, loss_curve_plot
from utilities import silhouette_analysis, gmm_analysis_plot, K_Means_inertia_plot, ICA_analysis,\
                       RP_analysis, hist_diagram, PCA_analysis_plot, Plot_2d, Plot_3d, LDA_analysis, gmm_plot
from sklearn.exceptions import ConvergenceWarning
from  warnings import simplefilter
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%%
# # Data Loading and Visualization
df=pd.read_csv("indian_liver_patient.csv")
df.head()

# # describing the data
print(df.info())
df.describe()

# # encoding the Gender attribute
df['Gender'].replace({'Male':1,'Female':0},inplace=True)
df['Dataset'].replace(2,0, inplace=True)

# # checking for missing values as per column
df.isna().sum()

# # checking the rows with the missing values
df[df['Albumin_and_Globulin_Ratio'].isna()]
df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())


# #  Data preprossesing
 
X = df.drop(['Dataset'], axis=1)
y = df['Dataset']
 
scaler1 = preprocessing.StandardScaler()
X_scale = scaler1.fit_transform(X)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

## Define scoring metric depending on it's binary or multiclass classification problem
if y.nunique()>2:   # multiclass case
    scoring_metric = 'f1_macro' 
else:
    scoring_metric = 'balanced_accuracy' 

#%% #### PART 1: Apply different clustering algorithms


# # Apply KMeans
def run_KMeans(X_scale, y):
    print('## APPLYING KMeans CLUSTERING ##')
    
    cluster_range = np.arange(2,11,1)
    silhouette_avg_list_Kmeans = silhouette_analysis(X_scale, cluster_range, 'KMeans')
    K_Means_inertia_plot(X_scale, cluster_range)
    
    best_n = 5
    kmeans = KMeans(n_clusters=best_n, random_state=1)
    
    start_time_train = time.time()
    kmeans.fit(X_scale)
    end_time_train = time.time()
    
    start_time_test = time.time()
    y_pred = kmeans.predict(X_scale)
    end_time_test = time.time()

    print('Model Training time(s):', end_time_train - start_time_train, 'Model Prediction time(s):', end_time_test - start_time_test)

#    print('homogeneity_score = ', homogeneity_score(y, y_pred))
    
#    where_0 = np.where(y_pred == 0)
#    where_1 = np.where(y_pred == 1)
#    if len(where_0[0]) > len(where_1[0]):
#        y_pred[where_0] =1
#        y_pred[where_1] =0
#        
#    contingency_mat = contingency_matrix(y, y_pred)
#    print('contingency_matrix: \n', contingency_mat)
#    hist_diagram(y_pred, 'KMeans')  # plot the histogram for predicted labels
#    
#    print('K-Means Inertia: ', kmeans.inertia_)
#    silh_result = silhouette_score(X_scale, kmeans.labels_)
#    print('K-Means Silhouette score: \n', silh_result)


run_KMeans(X_scale, y)

#%% #### PART 1: Apply different clustering algorithms

# # Apply GMM
def run_GMM(X_scale, y):
    
    print('## APPLYING GMM CLUSTERING ##')
    cluster_range = np.arange(2,11,1)
    gmm_analysis_plot(X_scale,cluster_range)
    silhouette_avg_list_gmm = silhouette_analysis(X_scale, cluster_range, 'gmm')
    
    best_n = 4
    gmm = GaussianMixture(best_n, n_init=1, random_state=1)
    
    start_time_train = time.time()
    gmm.fit(X_scale)
    end_time_train = time.time()
    
    start_time_test = time.time()
    y_pred = gmm.predict(X_scale)
    end_time_test = time.time()
    print('Model Training time(s):', end_time_train - start_time_train, 'Model Prediction time(s):', end_time_test - start_time_test)

    hist_diagram(y_pred, 'GMM')   # plot the histogram for predicted labels
    
    print('homogeneity_score = ', homogeneity_score(y, y_pred))
    
    where_0 = np.where(y_pred == 0)
    where_1 = np.where(y_pred == 1)
    if len(where_0[0]) > len(where_1[0]):
        y_pred[where_0] =1
        y_pred[where_1] =0
    contingency_mat = contingency_matrix(y, y_pred)
    print('contingency_matrix: \n', contingency_mat)
    
    silh_result = silhouette_score(X_scale, y_pred)
    print('K-Means Silhouette score: \n', silh_result)
    
run_GMM(X_scale, y)
    
#best_n = 6
#gmm = GaussianMixture(best_n, n_init=10, random_state=1)
#gmm.fit(X_scale)
#gmm_plot(gmm, X_scale)

#%%  #### PART 2: Apply dimension reduction algorithms

# # Apply PCA
print('## APPLYING PCA TO THE DATASET ##')
pca = PCA(random_state=1)
pca.fit(X_scale)
PCA_analysis_plot(pca)

# Visualize the data in 2d and 3d
Z_2d = PCA(n_components = 2).fit_transform(X_scale)
Plot_2d(Z_2d,y)

Z_3d = PCA(n_components = 3).fit_transform(X_scale)
Plot_3d(Z_3d,y)

best_n=6
best_pca = PCA(n_components = best_n)
best_X_pca = best_pca.fit_transform(X_scale)
x_reconstructed = best_pca.inverse_transform(best_X_pca)  # reconstruct
mse = np.mean((X_scale - x_reconstructed) ** 2)  # compute MSE
print('reconstruction MSE of PCA =', mse)



#%%  #### PART 2: Apply dimension reduction algorithms


# # Apply ICA
print('## APPLYING ICA TO THE DATASET ##')
component_range = np.arange(2,11,1)
ICA_analysis(X_scale, component_range)
# Visualize the data in 2d and 3d
Z_2d = FastICA(n_components = 2).fit_transform(X_scale)
Plot_2d(Z_2d,y)

Z_3d = FastICA(n_components = 3).fit_transform(X_scale)
Plot_3d(Z_3d,y)

best_n = 6
best_X_ica = FastICA(n_components = best_n).fit_transform(X_scale)



#%%  #### PART 2: Apply dimension reduction algorithms


# # Apply Random Projection
print('## APPLYING Random Projection TO THE DATASET ##')

component_range = range(1,11)
RP_analysis(X_scale, component_range)
# Visualize the data in 2d and 3d
Z_2d = GaussianRandomProjection(n_components = 2).fit_transform(X_scale)
Plot_2d(Z_2d,y)

Z_3d = GaussianRandomProjection(n_components = 3).fit_transform(X_scale)
Plot_3d(Z_3d,y)

best_n=6
best_rp = GaussianRandomProjection(n_components = best_n)
best_X_rp = best_rp.fit_transform(X_scale)

#%%  #### PART 2:Apply dimension reduction algorithms


# # Apply Tree classifier 
etc = ExtraTreesClassifier(n_estimators = 100)
etc = etc.fit(X_scale,y)
print(etc.feature_importances_)
model = SelectFromModel(etc, prefit=True)
X_Trees = model.transform(X)
print(model.get_support())



#%%  #### PART 2:Apply dimension reduction algorithms


# # Apply LDA
print('## APPLYING ICA TO THE DATASET ##')
X_transf = LDA_analysis(X_scale,y)


#%%  #### PART 3: Apply clustering algorithms on dimension reductioned data

# # Apply KMeans and GMM on PCA
print('## APPLYING KMeans on PCA DATA ##')

best_n=6
best_X_pca = PCA(n_components = best_n).fit_transform(X_scale)
run_KMeans(best_X_pca, y)

print('## APPLYING GMM on PCA DATA ##')
run_GMM(best_X_pca, y)

#%%  #### PART 3: Apply clustering algorithms on dimension reductioned data

# # Apply KMeans and GMM on ICA
print('## APPLYING KMeans on ICA DATA ##')
best_n=6
best_X_ica = FastICA(n_components = best_n).fit_transform(X_scale)
run_KMeans(best_X_ica, y)

print('## APPLYING GMM on ICA DATA ##')
run_GMM(best_X_ica, y)

#%%  #### PART 3: Apply clustering algorithms on dimension reductioned data

# # Apply KMeans and GMM on RP
print('## APPLYING KMeans on RP DATA ##')
best_n=6
best_X_rp = GaussianRandomProjection(n_components = best_n).fit_transform(X_scale)
run_KMeans(best_X_rp, y)

print('## APPLYING GMM on RP DATA ##')
run_GMM(best_X_rp, y)

#%%  #### PART 3: Apply clustering algorithms on dimension reductioned data

# # Apply KMeans and GMM on LDA
print('## APPLYING KMeans on LDA DATA ##')
X_transf = LDA_analysis(X_scale,y)
run_KMeans(X_transf, y)

print('## APPLYING GMM on LDA DATA ##')
run_GMM(X_transf, y)

#%%  #### PART 4: Ruuning neural network with Dimension Reductioned DATA

simplefilter("ignore", category=ConvergenceWarning)  ## Disable the not converge warning message

def run_neural_net(X_train, y_train, X_test, y_test, scoring_metric, cv):
    
    print("Starts to fit NN...")

    clf = MLPClassifier(hidden_layer_sizes= (20,10),  random_state=0, max_iter=1000, activation = 'relu', early_stopping = False)
    pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
    
    ## GridSearchCV
    
    param_grid = {'clf__learning_rate_init':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], 'clf__alpha': np.logspace(-3,3,8)}
    gscv_model, gscv_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv)
    
    ## Learning Curve
    train_scores, test_scores = learning_curve_plot(gscv_model, "NN", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)
    loss_curve_plot(gscv_model.best_estimator_['clf'])


print('## Running neural network on original data ##')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
run_neural_net(X_train, y_train, X_test, y_test, scoring_metric, cv)

#%%  #### PART 4: Ruuning neural network with Dimension Reductioned DATA

# # Running neural network on PCA data
print('## Running neural network on PCA data ##')
best_n=2
best_X_pca = PCA(n_components = best_n).fit_transform(X_scale)
X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(best_X_pca, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
run_neural_net(X_PCA_train, y_PCA_train, X_PCA_test, y_PCA_test, scoring_metric, cv)


#%%  #### PART 4: Ruuning neural network with Dimension Reductioned DATA

# # Running neural network on ICA data
print('## Running neural network on ICA data ##')
best_n=6
best_X_ica = FastICA(n_components = best_n).fit_transform(X_scale)
X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(best_X_ica, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
run_neural_net(X_PCA_train, y_PCA_train, X_PCA_test, y_PCA_test, scoring_metric, cv)


#%%  #### PART 4: Ruuning neural network with Dimension Reductioned DATA

# # Running neural network on RP data
print('## Running neural network on RP data ##')
best_n=6
best_X_rp = GaussianRandomProjection(n_components = best_n).fit_transform(X_scale)
X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(best_X_rp, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
run_neural_net(X_PCA_train, y_PCA_train, X_PCA_test, y_PCA_test, scoring_metric, cv)


#%%  #### PART 4: Ruuning neural network with Dimension Reductioned DATA

# # Running neural network on LDA data
print('## Running neural network on LDA data ##')
X_transf = LDA_analysis(X_scale,y)
X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(X_transf, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
run_neural_net(X_PCA_train, y_PCA_train, X_PCA_test, y_PCA_test, scoring_metric, cv)

#%%  #### PART 5: Ruuning neural network with data + clustering as new feature

# # Running neural network with KMeans clustered data
print('## Running neural network with data + KMeans clustering ##')
best_n=2
k_means_clustering = KMeans(n_clusters=best_n, random_state=1)
X_KMeans = np.append(X, k_means_clustering.fit_transform(X_scale), 1)
X_KMeans_train, X_KMeans_test, y_KMeans_train, y_KMeans_test = train_test_split(X_KMeans, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
run_neural_net(X_KMeans_train, y_KMeans_train, X_KMeans_test, y_KMeans_test, scoring_metric, cv)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
best_n=2
k_means_clustering = KMeans(n_clusters=best_n, random_state=1)
k_means_clustering.fit(X_train)
X_KMeans_train = np.append(X_train, k_means_clustering.transform(X_train), 1)
X_KMeans_test = np.append(X_test, k_means_clustering.transform(X_test), 1)
run_neural_net(X_KMeans_train, y_train, X_KMeans_test, y_test, scoring_metric, cv)


#%%  #### PART 5: Ruuning neural network with clustering

# # Running neural network with GMM clustered data
print('## Running neural network with data + GMM clustering ##')
best_n=2
gmm = GaussianMixture(best_n, n_init=10, random_state=1)
gmm.fit(X_scale)
X_GMM = np.append(X, gmm.predict_proba(X_scale), 1)
X_GMM_train, X_GMM_test, y_GMM_train, y_GMM_test = train_test_split(X_GMM, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
run_neural_net(X_GMM_train, y_GMM_train, X_GMM_test, y_GMM_test, scoring_metric, cv)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
best_n=2
gmm = GaussianMixture(best_n, n_init=10, random_state=1)
gmm.fit(X_train)
X_GMM_train = np.append(X_train, gmm.predict_proba(X_train), 1)
X_GMM_test = np.append(X_test, gmm.predict_proba(X_test), 1)
run_neural_net(X_GMM_train, y_train, X_GMM_test, y_test, scoring_metric, cv)