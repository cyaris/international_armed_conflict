import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedShuffleSplit
import optunity
import graphviz
from sklearn.tree import export_graphviz

def y_pred_inverse(predict_proba_statement):
    predicted_plea = []
    y_pred = []
    for i in predict_proba_statement:
        predicted_plea.append(i[0])
        y_pred.append(i[1])
    y_pred = np.array(y_pred).reshape(-1, 1)
    return y_pred

# def plot_validation_curve(estimator, param_name, param_range, title, X, y):
#     train_scores, validation_scores = validation_curve(
#     estimator, X, y, param_name, param_range = param_range,
#     cv = 5, scoring = 'neg_mean_squared_error', n_jobs = 1)
#     train_rmse = np.mean(np.sqrt(-train_scores), axis = 1)
#     validation_rmse = np.mean(np.sqrt(-validation_scores), axis = 1)

#     plt.figure(figsize = (9, 5))
#     plt.title(title)
#     plt.xlabel(param_name)
#     plt.ylabel('RMSE')
#     plt.semilogx(param_range, train_rmse, label = 'Training RMSE',
#                  color = 'darkorange', lw = 2, marker = '.', markerfacecolor = 'white',
#                  markersize = 8, markeredgecolor = 'black', markeredgewidth = .5)
#     plt.semilogx(param_range, validation_rmse, label = 'Validation RMSE',
#                  color = 'navy', lw = 2, marker = '.', markerfacecolor = 'white',
#                  markersize = 8, markeredgecolor = 'black', markeredgewidth = .5)
#     plt.legend(loc = 'best')

def plot_validation_curve_log(estimator, param_name, param_range, title, X, y):
    train_scores, validation_scores = validation_curve(
    estimator, X, y, param_name, param_range = param_range,
    cv = 5, scoring = 'roc_auc', n_jobs = 1)
    train_rmse = np.mean(train_scores, axis = 1)
    validation_rmse = np.mean(validation_scores, axis = 1)

    plt.figure(figsize = (9, 5))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('ROC/AUC')
    plt.semilogx(param_range, train_rmse, label = 'Training ROC/AUC',
                 color = 'darkorange', lw = 2, marker = '.', markerfacecolor = 'white',
                 markersize = 8, markeredgecolor = 'black', markeredgewidth = .5)
    plt.semilogx(param_range, validation_rmse, label = 'Validation ROC/AUC',
                 color = 'navy', lw = 2, marker = '.', markerfacecolor = 'white',
                 markersize = 8, markeredgecolor = 'black', markeredgewidth = .5)
    plt.legend(loc = 'best')
    
def plot_validation_curve_reg(estimator, param_name, param_range, title, X, y):
    train_scores, validation_scores = validation_curve(
    estimator, X, y, param_name, param_range = param_range,
    cv = 5, scoring = 'roc_auc', n_jobs = 1)
    train_rmse = np.mean(train_scores, axis = 1)
    validation_rmse = np.mean(validation_scores, axis = 1)

    plt.figure(figsize = (9, 5))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('ROC/AUC')
    plt.plot(param_range, train_rmse, label = 'Training ROC/AUC',
             color = 'darkorange', lw = 2, marker = '.', markerfacecolor = 'white',
             markersize = 8, markeredgecolor = 'black', markeredgewidth = .5)
    plt.plot(param_range, validation_rmse, label = 'Validation ROC/AUC',
             color = 'navy', lw = 2, marker = '.', markerfacecolor = 'white',
             markersize = 8, markeredgecolor = 'black', markeredgewidth = .5)
    plt.legend(loc = 'best')

def plot_learning_curve(estimator, title, X, y, train_sizes = np.linspace(.1, 1.0, 8)):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv = 5, shuffle = True, scoring = 'neg_mean_squared_error',
        n_jobs = 1, train_sizes = train_sizes)
    train_rmse = np.mean(np.sqrt(-train_scores), axis = 1)
    validation_rmse = np.mean(np.sqrt(-validation_scores), axis = 1)

    plt.figure(figsize = (9, 5))
    plt.title(title)
    plt.xlabel('Number of Training Observations')
    plt.ylabel('RMSE')
    plt.plot(train_sizes, train_rmse, color = 'darkorange', lw = 2, marker = '.', markerfacecolor = 'white',
             markersize = 8, markeredgecolor = 'black', markeredgewidth = .5,
             label = 'Training Score')
    plt.plot(train_sizes, validation_rmse, color = 'navy', lw = 2, marker = '.', markerfacecolor = 'white',
             markersize = 8, markeredgecolor = 'black', markeredgewidth = .5,
             label = 'Validation Score')
    plt.legend(loc = "best")
    
def plot_decision_tree(clf, feature_names, class_names):
    export_graphviz(clf, out_file="dtree.dot", feature_names = feature_names, class_names=class_names, filled = True, impurity = False)
    with open("dtree.dot") as f:
        dot_graph = f.read()
    return graphviz.Source(dot_graph)

def train_and_calibrate_cv(models, X_train_scaled, y_train, cv = 5):
    model_scores = {}
    for i, model in enumerate(models):
        model_fold_scores = []
        skf = StratifiedShuffleSplit(n_splits = cv, test_size = 0.2, random_state = 101)
        for train_index, validation_index in skf.split(X_train_scaled, y_train):
            X_train_scaled_v = df[feature_names].iloc[train_index]
            X_validation_scaled_v = df[feature_names].iloc[validation_index]
            y_train_v = df['found_guilty'].iloc[train_index]
            y_validation_v = df['found_guilty'].iloc[validation_index]
            model.fit(X_train_scaled_v, y_train_v)
            try:
                y_pred_proba_v = y_pred_inverse(model.predict_proba(X_validation_scaled_v))
                model_fold_scores.append(metrics.roc_auc_score(y_validation_v, y_pred_proba_v))
            except:
                y_pred_v = model.predict(X_validation_scaled_v)
                model_fold_scores.append(metrics.roc_auc_score(y_validation_v, y_pred_v))
        print(model, "\nAverage ROC/AUC Score: ", pd.Series(model_fold_scores).mean())
        model_scores[model] = pd.Series(model_fold_scores).mean()
    return model_scores

def cm_val_scaled(model, predict_x, true_y, threshold = 0.5):
   # Predict class 1 if probability of being in class 1 is greater than threshold
   # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(predict_x)[:, 1] >= threshold)
    confusion = metrics.confusion_matrix(true_y, y_predict)
    plt.figure(dpi = 80)
    sns.heatmap(confusion, cmap = plt.cm.Blues, annot = True, square = True, fmt = 'd',
                xticklabels=['Plead Guilty', 'Found Guilty'],
                yticklabels=['Plead Guilty', 'Found Guilty']);
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    
def cm_val(model, threshold = 0.5):
   # Predict class 1 if probability of being in class 1 is greater than threshold
   # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    confusion = metrics.confusion_matrix(y_test, y_predict)
    plt.figure(dpi = 80)
    sns.heatmap(confusion, cmap = plt.cm.Blues, annot = True, square = True, fmt = 'd',
                xticklabels=['Plead Guilty', 'Found Guilty'],
                yticklabels=['Plead Guilty', 'Found Guilty']);
    plt.xlabel('Prediction')
    plt.ylabel('Actual')