# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 06:57:01 2022

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

## Data Loading and Visualization
df=pd.read_csv("winequalityN.csv")
df.head()

##Describing the data
df.describe()
plt.figure()
sns.countplot(df['type'])
plt.figure()
sns.countplot(df['quality'])
df['quality'].value_counts()

## Mapping values of target variable quality to 'low', 'medium' and 'high' categories for classification
df['quality']=df['quality'].map({3:'low', 4:'low', 5:'medium', 6:'medium', 7:'medium', 8:'high', 9:'high'})
df['quality']=df['quality'].map({'low':0,'medium':1, 'high':2})
df.sample(5)

##Checking for missing values as per column
df.isna().sum()

##Fill the missing values
for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())


#Removing outliers in residual sugar
lower = df['residual sugar'].mean()-3*df['residual sugar'].std()
upper = df['residual sugar'].mean()+3*df['residual sugar'].std()
df = df[(df['residual sugar']>lower) & (df['residual sugar']<upper)]

#Removing outliers in free sulfur dioxide
lower = df['free sulfur dioxide'].mean()-3*df['free sulfur dioxide'].std()
upper = df['free sulfur dioxide'].mean()+3*df['free sulfur dioxide'].std()
df = df[(df['free sulfur dioxide']>lower) & (df['free sulfur dioxide']<upper)]

#Removing outliers in total sulfur dioxide
lower = df['total sulfur dioxide'].mean()-3*df['total sulfur dioxide'].std()
upper = df['total sulfur dioxide'].mean()+3*df['total sulfur dioxide'].std()
df = df[(df['total sulfur dioxide']>lower) & (df['total sulfur dioxide']<upper)]


##encoding the wine type attribute
le = preprocessing.LabelEncoder()
df['type'] = le.fit_transform(df['type'])
df = df.reset_index()
df.head()

 
X = df.drop(['quality'], axis=1)
y = df['quality']


scaler1 = preprocessing.StandardScaler()
X_scale = scaler1.fit_transform(X)

print("Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

scaler2 = preprocessing.StandardScaler()
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

#%% #### PART 1: Apply different clustering algorithms


# # Apply KMeans
def run_KMeans(X_scale, y):
    print('## APPLYING KMeans CLUSTERING ##')
    
    cluster_range = np.arange(2,11,1)
    silhouette_avg_list_Kmeans = silhouette_analysis(X_scale, cluster_range, 'KMeans')
    K_Means_inertia_plot(X_scale, cluster_range)
    
    best_n = 3
    kmeans = KMeans(n_clusters=best_n, random_state=1)

    start_time_train = time.time()
    kmeans.fit(X_scale)
    end_time_train = time.time()
    
    start_time_test = time.time()
    y_pred = kmeans.predict(X_scale)
    end_time_test = time.time()

    print('Model Training time(s):', end_time_train - start_time_train, 'Model Prediction time(s):', end_time_test - start_time_test)
    
    hist_diagram(y_pred, 'KMeans')  # plot the histogram for predicted labels
    print('K-Means Inertia: ', kmeans.inertia_)
    silh_result = silhouette_score(X_scale, kmeans.labels_)
    print('K-Means Silhouette score: \n', silh_result)

run_KMeans(X_scale, y)

#%% #### PART 1: Apply different clustering algorithms

# # Apply GMM
def run_GMM(X_scale, y):
    
    print('## APPLYING GMM CLUSTERING ##')
    cluster_range = np.arange(2,11,1)
    gmm_analysis_plot(X_scale,cluster_range)
    #silhouette_avg_list_gmm = silhouette_analysis(X_scale, cluster_range, 'gmm')
    
    best_n = 7
    gmm = GaussianMixture(best_n, n_init=1, random_state=1)
    
    start_time_train = time.time()
    gmm.fit(X_scale)
    end_time_train = time.time()
    
    start_time_test = time.time()
    y_pred = gmm.predict(X_scale)
    end_time_test = time.time()
    print('Model Training time(s):', end_time_train - start_time_train, 'Model Prediction time(s):', end_time_test - start_time_test)
    
    hist_diagram(y_pred, 'GMM')   # plot the histogram for predicted labels
    
    
run_GMM(X_scale, y)
    
#best_n = 7
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

best_n=8
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

best_n = 8
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

best_n=8
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

