# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 08:51:21 2022

@author: cheng164
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_roc_curve, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from yellowbrick.classifier.rocauc import roc_auc

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples
import matplotlib.cm as cm
from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
import itertools
import scipy.stats
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import GaussianRandomProjection
from scipy.linalg import pinv
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%%
## Citation: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
def tree_accuracy_vs_alpha(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )
    
     # Remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node. 
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    

def validation_curve_plot(model, X_train, y_train, clf_name, param_name, param_range, scoring, cv):
    
    train_scores, test_scores = validation_curve(model, X_train, y_train, param_range=param_range, param_name=param_name, scoring =scoring, cv=cv)
    
    plt.figure()
    
    if clf_name == "SVM" or clf_name == "NN" or param_name =='clf__learning_rate':
        plt.semilogx(param_range, np.mean(train_scores, axis=1), label='Train Score')
        plt.semilogx(param_range, np.mean(test_scores, axis=1), label='CV Score') 
    else:       
        plt.plot(param_range, np.mean(train_scores, axis=1), label='Train Score')
        plt.plot(param_range, np.mean(test_scores, axis=1), label='CV Score')
    
    plt.legend(loc="best")
    plt.title("Validation Curve with {} ({})".format(param_name, clf_name))
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.grid()
    plt.show()
    #plt.savefig("results/{}_model_complexity_{}_{}.png".format(clf_name, dataset_name, param_name))
    
    

def GridSearchCV_result(model, X_train, y_train, X_test, y_test, param_grid, scoring, cv , classes=None):

    gscv_clf = GridSearchCV(estimator = model, param_grid=param_grid, scoring=scoring, cv=cv)
    
    start_time_train = time.time()
    gscv_clf.fit(X_train, y_train)
    end_time_train = time.time()
    
    start_time_test = time.time()
    y_pred = gscv_clf.predict(X_test)
    end_time_test = time.time()
    print('Model Training time(s):', end_time_train - start_time_train, 'Model Prediction time(s):', end_time_test - start_time_test)
    
    print("Best Hyperparameters are: {} ; Best Mean CV Score: {}".format(gscv_clf.best_params_ , gscv_clf.best_score_))
    
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    p.set(xlabel='Predictable label', ylabel='Actual label', title = 'Confusion Matrix')


    # ROC & AUC
    y_pred_probas = gscv_clf.predict_proba(X_test)
    
    if y_test.nunique()>2:   # multiclass case
        auc = roc_auc_score(y_test, y_pred_probas, multi_class='ovo', average='macro') 
        plt.figure()
        roc_auc(gscv_clf, X_train, y_train, X_test, y_test, classes= classes) 
    else:
        auc = roc_auc_score(y_test, y_pred_probas[:, 1])
        plot_roc_curve(gscv_clf, X_test, y_test)  # plot ROC curves 

    if scoring == 'balanced_accuracy':
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        print("Balanced Accuracy of the best model:", balanced_accuracy)
        
    classifier_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the best model:", classifier_accuracy)
    print("Model AUC:", auc)
    print('Classification Report: \n', classification_report(y_test, y_pred))
    
    return gscv_clf, gscv_clf.best_score_ 

 
 
def learning_curve_plot(model, clf_name, X_train, y_train, train_size_pct, scoring, cv):

    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_size_pct, scoring=scoring, cv=cv)
    
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve for {}".format(clf_name))
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()
    plt.show()
    
    
    return train_scores, test_scores
    

def loss_curve_plot(mlp_clf):
    plt.figure()
    plt.plot(mlp_clf.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    

#%%

def K_Means_inertia_plot(X, cluster_range):
    inertia_ls = []
    for num_cluster in cluster_range:
        clusterer = KMeans(n_clusters=num_cluster, random_state=1)
        cluster_labels = clusterer.fit(X)
        inertia_ls.append(clusterer.inertia_)
    plt.figure()
    plt.plot(cluster_range, np.array(inertia_ls))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters (Kmeans)', fontsize=20)
    plt.show()
    
    # Use KElbowVisualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(np.amin(cluster_range), np.amax(cluster_range)))
    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure 
    


def silhouette_analysis(X, cluster_range, algorithm):
    ### citation: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    
    silhouette_avg_list= []
    
    for num_cluster in cluster_range:
        plt.figure()
        ax1=plt.gca()
	   
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (num_cluster + 1) * 10])

        if algorithm == 'KMeans':
            clusterer = KMeans(n_clusters=num_cluster, random_state=1)
  
        else:
            clusterer= GaussianMixture(n_components=num_cluster, n_init=10, random_state=1)
            
        cluster_labels = clusterer.fit_predict(X)    
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        print("For n_clusters = ", num_cluster,"The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(num_cluster):
	        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

	        ith_cluster_silhouette_values.sort()

	        size_cluster_i = ith_cluster_silhouette_values.shape[0]
	        y_upper = y_lower + size_cluster_i

	        color = cm.nipy_spectral(float(i) / num_cluster)
	        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

	        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

	        y_lower = y_upper + 10 

        ax1.set_title("The silhouette plot for number of clusters="+str(num_cluster))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  
        ax1.set
        
        
    # Plot Average silhouette Score vs. cluster_range      
    plt.figure()
    plt.plot(cluster_range, silhouette_avg_list)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average silhouette Score')
    plt.title('Avg silhouette Score vs. Number of Clusters', fontsize=20)
    plt.show()
        
    return silhouette_avg_list 



def gmm_analysis_plot(X, cluster_range):
    bics=[]
    ll_scores = []
    
    for n in cluster_range:
        gmm=GaussianMixture(n, n_init=10, random_state=1).fit(X) 
        bics.append(gmm.bic(X))
        ll_scores.append(gmm.score(X))

    plt.figure()
    plt.plot(cluster_range, bics, label='BIC')
    plt.title("BIC Scores", fontsize=20)
    plt.xticks(cluster_range)
    plt.xlabel("N. of component")
    plt.ylabel("BIC Score")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(cluster_range, np.gradient(bics), label='BIC')
    plt.title("Gradient of BIC Scores", fontsize=20)
    plt.xticks(cluster_range)
    plt.xlabel("N. of component")
    plt.ylabel("grad(BIC)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(cluster_range, ll_scores, label='log-likelihood')
    plt.title("average log-likelihood", fontsize=20)
    plt.xticks(cluster_range)
    plt.xlabel("N. of component")
    plt.ylabel("Avg log-likelihood")
    plt.legend()
    plt.show()
    

def gmm_plot(clf, X):

    # Plot the winner
    color_iter = itertools.cycle(['red', 'blue', 'green', 'black', 'yellow', 'turquoise', 'khaki', 'pink', 'moccasin', 'olive', 'coral'])
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 3], X[Y_ == i, 5], 0.8, color=color)
    
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    
    plt.xticks(())
    plt.yticks(())
    plt.title(
        f"Selected GMM: {clf.covariance_type} model, "
        f"{clf.n_components} components"
    )
    plt.subplots_adjust(hspace=0.35, bottom=0.02)
    plt.show()    
    

def hist_diagram(labels, algorithm):

    plt.figure()
    n = len(np.unique(labels))
    plt.hist(labels, bins=np.arange(0, n+1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, n))
    plt.xlabel('Cluster')
    plt.ylabel('Samples per Cluster')
    plt.title('Distribution of data per cluster for '+ str(algorithm))
    plt.grid()



#%%

def PCA_analysis_plot(pca):
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), pca.explained_variance_ratio_, label='var')
    ax1.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), np.cumsum(pca.explained_variance_ratio_), label='cum var')
    # ax1.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1, 2))
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(loc=3, fontsize=15)
    plt.grid()

    ax2 = ax1.twinx()
    ax2.plot(list(range(1, len(pca.singular_values_)+1)), pca.singular_values_, 'm-', label='eigenvalues')
    ax2.set_ylabel('Eigenvalues', color='m')
    ax2.tick_params('y', colors='m')
    ax2.legend(loc=7, fontsize=15)
    plt.title("PCA Explained Variance and Eigenvalues")
    fig.tight_layout()
    plt.show()



def Plot_2d(Z,y): 
    ## CITATION: https://towardsdatascience.com/dimensionality-reduction-toolbox-in-python-9a18995927cd
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'turquoise', 'khaki', 'pink', 'moccasin', 'olive', 'coral']
    plt.figure()
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[0], alpha=0.5,label='0')
        elif y[i] == 1:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[1], alpha=0.5,label='1')
        elif y[i] == 2:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[2], alpha=0.5,label='2')
        elif y[i] == 3:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[3], alpha=0.5,label='3')
        elif y[i] == 4:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[4], alpha=0.5,label='4')
        elif y[i] == 5:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[5], alpha=0.5,label='5')    
        elif y[i] == 6:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[6], alpha=0.5,label='6')   
        elif y[i] == 7:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[7], alpha=0.5,label='7') 
        elif y[i] == 8:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[8], alpha=0.5,label='8')  
        elif y[i] == 9:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[9], alpha=0.5,label='9')  
        elif y[i] == 10:
            plt.scatter(Z[i, 1], Z[i, 0], color = colors[10], alpha=0.5,label='10')  
    plt.show()
    



def Plot_3d(Z,y):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'turquoise', 'khaki', 'pink', 'moccasin', 'olive', 'coral']

    for i in range(len(y)):
        if y[i] == 0:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = colors[0], marker='o')
        elif y[i] == 1:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = colors[1], marker='o')
        elif y[i] == 2:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = colors[2], marker='o')
        elif y[i] == 3:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = colors[3], marker='o')
        elif y[i] == 4:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = colors[4], marker='o')
        elif y[i] == 5:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = colors[5], marker='o')
        elif y[i] == 6:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = colors[6], marker='o')
    plt.show()



def ICA_analysis(X, component_range):
    arr_kurt = []
    arr_mse = []
    for i in component_range:
        ICA = FastICA(n_components = i)
        X_indep =ICA.fit_transform(X)
        kurt = scipy.stats.kurtosis(X_indep)
        arr_kurt.append(np.mean(kurt))
        
        x_reconstructed = ICA.inverse_transform(X_indep)  # reconstruct
        mse = np.mean((X - x_reconstructed) ** 2)  # compute MSE
        arr_mse.append(mse)
        
    plt.figure()
    plt.plot(component_range, np.array(arr_kurt))
    plt.xlabel('Number of Components')
    plt.ylabel('Kurtosis Value')
    plt.title('Mean Kurtosis Value vs. Number of Components for ICA')
    plt.show()
    
    plt.figure()
    plt.plot(component_range, np.array(arr_mse))
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Square Error')
    plt.title('Inverse transformation MSE vs. Number of Components for ICA')
    plt.show()


def RP_analysis(X, component_range):
    arr_mse = []
    arr_dist = []
    for i in component_range:
        rp = GaussianRandomProjection(n_components=i)
        X_rp = rp.fit(X)
        p = pinv(X_rp.components_)
        w = X_rp.components_
        reconstructed = ((p@w)@(X.T)).T 
        arr_mse.append(mean_squared_error(X,reconstructed))

        X_transf = rp.fit_transform(X)
        dist_raw = euclidean_distances(X)
        dist_transform = euclidean_distances(X_transf)
        abs_diff_gauss = abs(dist_raw - dist_transform) 
        arr_dist.append(np.mean(abs_diff_gauss.flatten()))
        
    plt.figure()
    plt.plot(component_range,np.array(arr_mse))
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error', fontsize=20)
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(component_range,np.array(arr_dist))
    plt.xlabel('Number of Components')
    plt.ylabel('Mean absolute difference')
    plt.title('Visualization of absolute differences',  fontsize=20)
    plt.grid()
    plt.show()
    
    
    
def LDA_analysis(X,y):
    # citation: https://www.statology.org/linear-discriminant-analysis-in-python/
    
    lda = LDA()
    X_transf = lda.fit_transform(X, y)
    print('Shape of transformed X =', X_transf.shape)
    
    # plot explained_variance_ratio_ 
    plt.figure()
    plt.plot(np.arange(1, lda.explained_variance_ratio_.size + 1), lda.explained_variance_ratio_, 'o-',label='var')
    plt.plot(np.arange(1, lda.explained_variance_ratio_.size + 1), np.cumsum(lda.explained_variance_ratio_), 'o-', label='cum var')
    # ax1.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1, 2))
    plt.xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    plt.ylabel('Explained Variance Ratio')
    plt.legend()


    
    ## plot the 2D projection plot
    plt.figure()
    color_ls = ['red', 'blue', 'green', 'black', 'yellow', 'turquoise', 'khaki', 'pink', 'moccasin', 'olive', 'coral']
    target_names = [",".join(item) for item in y.unique().astype(str)]
    colors = color_ls[0:len(target_names)]
    
    if X_transf.shape[1]<2:
        for color, i, target_name in zip(colors, [0, 1], target_names):
            if i==0:
                plt.scatter(X_transf[y == i], np.zeros_like(X_transf[y == i])-0.01 , alpha=.5, color=color, label=target_name)
            if i==1:
                plt.scatter(X_transf[y == i], np.zeros_like(X_transf[y == i])+0.01 , alpha=.5, color=color, label=target_name)
        
        plt.ylim([-1, 1]) 
        plt.xlabel('LDA 1st dimension')
        plt.ylabel('LDA 2nd dimension')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.show()
    
    else:
        for color, i, target_name in zip(colors, list(range(0,len(target_names))), target_names):
            plt.scatter(X_transf[y == i, 0], X_transf[y == i, 1], alpha=.8, color=color, label=target_name)        

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.xlabel('LDA 1st dimension')
        plt.ylabel('LDA 2nd dimension')
        plt.show()
    
    return X_transf