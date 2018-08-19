# In[1]:
import numpy as np
import pandas as pd
# In[2]:
import matplotlib
import matplotlib.pyplot as plt
# In[3]:
from sklearn.datasets import fetch_mldata
# In[4]:
mnist = fetch_mldata('MNIST ORIGINAL')
# In[5]:
mnist
# In[6]:
mnist.data
# In[7]:
mnist.data.shape
# In[8]:
X = mnist.data
# In[9]:
Y = mnist.target
# In[10]:
some_digit = X[36000]
# In[11]:
some_digit_image = some_digit.reshape(28,28)
# In[12]:
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest')
# In[13]:
Y[36000]
# In[14]:
X_train, Y_train, X_test, Y_test = X[:60000], Y[:60000], X[60000:], Y[60000:]
# In[15]:
random_index = np.random.permutation(60000)
# In[16]:
X_train, Y_train = X[random_index], Y[random_index]
# In[17]:
from sklearn.linear_model import SGDClassifier
# In[18]:
sgd_clf = SGDClassifier(random_state = 42)
# In[19]:
Y_train_5 = Y_train == 5
# In[20]:
sgd_clf.fit(X_train, Y_train)
# In[21]:
sgd_clf.predict([some_digit])
# In[22]:
from sklearn.model_selection import StratifiedKFold
# In[23]:
skfold = StratifiedKFold(n_splits = 3, random_state = 42)
# In[24]:
from sklearn.base import clone
for train_index,test_index in skfold.split(X_train, Y_train_5):
    sgd_clone = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    Y_train_folds = Y_train_5[train_index]
    X_test_fold = X_train[test_index]
    Y_test_fold = Y_train_5[test_index]
    sgd_clone.fit(X_train_folds,Y_train_folds)
    predictions = sgd_clone.predict(X_test_fold)
    num_correct = sum(predictions == Y_test_fold)
    score = num_correct / len(predictions)
    print(score)
# In[25]:
from sklearn.model_selection import cross_val_score
# In[26]:
score = cross_val_score(sgd_clf, X_train, Y_train_5, scoring = "accuracy", cv = 3)
# In[27]:
score
# In[28]:
from sklearn.metrics import confusion_matrix
# In[29]:
from sklearn.model_selection import cross_val_predict
# In[30]:
prediction = cross_val_predict(sgd_clf, X_train, Y_train_5, cv = 3)
# In[31]:
conf_mat = confusion_matrix(Y_train_5, prediction)
# In[32]:
conf_mat
# In[33]:
conf_mat1 = confusion_matrix(Y_train_5, Y_train_5)
# In[34]:
conf_mat1
# In[35]:
from sklearn.metrics import precision_score
# In[36]:
from sklearn.metrics import recall_score
# In[37]:
from sklearn.metrics import accuracy_score
# In[38]:
accuracy = accuracy_score(Y_train_5, prediction)
# In[39]:
accuracy
# In[40]:
precisionscore = precision_score(Y_train_5, prediction)
# In[41]:
precisionscore
# In[42]:
recallscore = recall_score(Y_train_5, prediction)
# In[43]:
recallscore
# In[44]:
from sklearn.metrics import f1_score
# In[45]:
f1score = f1_score(Y_train_5, prediction)
# In[46]:
f1score
# In[47]:
Y_score = sgd_clf.decision_function([some_digit])
# In[48]:
Y_score
# In[49]:
threshold = 0
# In[50]:
Y_score > threshold
# In[51]:
threshold = 15000
# In[52]:
Y_score > threshold
# In[53]:
Y_score = cross_val_predict(sgd_clf, X_train, Y_train_5, cv = 3, method = 'decision_function')
# In[54]:
Y_score
# In[55]:
from sklearn.metrics import precision_recall_curve
# In[56]:
precision, recall, threshold = precision_recall_curve(Y_train_5, Y_score)
# In[57]:
def plot_precision_recall_vs_threshold(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], 'b-', label = 'Precision')
    plt.plot(threshold, recall[:-1], 'g', label = 'Recall')
    plt.xlabel("Threshold")
    plt.legend(loc = 'center left')
    plt.ylim(0, 1)
plot_precision_recall_vs_threshold(precision, recall, threshold)
plt.show()
# In[58]:
threshold = 60000
Y_score_90 = (Y_score > threshold)
preci_score = precision_score(Y_train_5, Y_score_90)
print(preci_score)
# In[59]:
recal_score = recall_score(Y_train_5, Y_score_90)
print(recal_score)
# In[60]:
def precision_vs_recall(precision, recall):
    plt.plot(recall, precision, 'b-', linewidth = 2)
    plt.xlabel('Recall', color = 'r')
    plt.ylabel('Precision', color = 'r')
precision_vs_recall(precision, recall)
plt.show()
# In[61]:
from sklearn.base import BaseEstimator, TransformerMixin
# In[62]:
class MySGDClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0, random_state=42):
        self.threshold = threshold
        self.random_state = random_state
        self.classifier = SGDClassifier(random_state = random_state)
    def fit(self,X,y):
        self.classifier.fit(X,y)
        return self
    def predict(self,x):
        Y_score_values = self.classifier.decision_function(x)
        Y_values = Y_score_values > self.threshold
        return Y_values
# In[63]:
my_sgd_clf = MySGDClassifier(threshold = 60000, random_state = 42)
# In[64]:
my_sgd_clf.fit(X_train, Y_train_5)
# In[65]:
Y_predicts = my_sgd_clf.predict(X_train)
# In[66]:
precision_score(Y_train_5, Y_predicts)
# In[67]:
recall_score(Y_train_5, Y_predicts)
# In[68]:
from sklearn.metrics import roc_curve, roc_auc_score
# In[69]:
fpr, tpr, threshold = roc_curve(Y_train_5, Y_score)
# In[70]:
def plot_tpr_vs_fpr(fpr, tpr, label = None):
    plt.plot(fpr, tpr, 'b-', linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', color = 'r')
    plt.ylabel('True Positive Rate', color = 'r')
plot_tpr_vs_fpr(fpr, tpr)
plt.show()
# In[71]:
sgd_auc = roc_auc_score(Y_train_5, Y_score)
# In[72]:
sgd_auc
# In[73]:
from sklearn.ensemble import RandomForestClassifier
# In[74]:
random_clf = RandomForestClassifier(random_state = 42)
# In[75]:
random_clf.fit(X_train, Y_train_5)
# In[76]:
Y_score_random = random_clf.predict_proba([some_digit])
# In[77]:
Y_score_random
# In[78]:
from sklearn.model_selection import cross_val_predict
# In[79]:
Y_score_random = cross_val_predict(random_clf, X_train, Y_train_5, cv = 3, method = 'predict_proba')
# In[80]:
Y_score_random
# In[81]:
Y_score_random_pos = Y_score_random[:, 1]
# In[82]:
fpr_random, tpr_random, threshold = roc_curve(Y_train_5, Y_score_random_pos)
# In[83]:
plt.plot(fpr, tpr, 'g:', label = 'SGD')
plot_tpr_vs_fpr(fpr_random, tpr_random, label = 'Random Forest Classifier')
plt.legend(loc = 'lower right')
plt.show()
# In[84]:
randomforest_auc = roc_auc_score(Y_train_5, Y_score_random_pos)
# In[85]:
randomforest_auc
# In[86]:
sgd_clf.fit(X_train, Y_train)
# In[87]:
sgd_clf.predict([some_digit])
# In[88]:
sgd_clf.decision_function([some_digit])
# In[89]:
from sklearn.multiclass import OneVsOneClassifier
# In[90]:
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state = 42))
# In[91]:
ovo_clf.fit(X_train, Y_train)
# In[92]:
ovo_clf.predict([some_digit])
# In[93]:
len(ovo_clf.estimators_)
# In[94]:
random_clf.fit(X_train, Y_train)
# In[95]:
random_clf.predict_proba([some_digit])
# In[96]:
score = cross_val_score(sgd_clf, X_train, Y_train, cv = 3, scoring = 'accuracy')
# In[97]:
score
# In[98]:
from sklearn.preprocessing import StandardScaler
# In[99]:
scaler = StandardScaler()
# In[100]:
X_train_64 = X_train.astype(np.float64)
# In[101]:
X_train_scaler = scaler.fit_transform(X_train_64)
# In[102]:
score = cross_val_score(sgd_clf, X_train_scaler, Y_train, cv = 3, scoring = 'accuracy')
# In[103]:
score
# In[104]:
score = cross_val_predict(sgd_clf, X_train_scaler, Y_train, cv = 3)
# In[105]:
score
# In[106]:
conf_mat = confusion_matrix(Y_train, score)
# In[107]:
conf_mat
# In[108]:
plt.matshow(conf_mat, cmap = plt.cm.gray)
# In[109]:
from sklearn.neighbors import KNeighborsClassifier
# In[110]:
Y_train_large = Y_train > 7
# In[111]:
Y_train_large
# In[112]:
Y_train_odd = Y_train % 2 != 0
# In[121]:
Y_train_odd
# In[113]:
Y_train_multi = np.c_[Y_train_large, Y_train_odd]
# In[115]:
Y_train_multi
# In[116]:
knc_clf = KNeighborsClassifier()
# In[117]:
knc_clf.fit(X_train, Y_train_multi)
# In[118]:
knc_clf.predict([some_digit])
# In[119]:
noise = np.random.randint(0, 100, (len(X_test),784))
# In[120]:
X_test_mod = X_test + noise
# In[121]:
Y_train_mod = X_train
# In[122]:
noise = np.random.randint(0, 100, (len(X_train),784))
# In[123]:
X_train_mod = X_train + noise
# In[124]:
knc_clf.fit(X_train_mod, Y_train_mod)
# In[125]:
Y_test_mod = X_test
# In[126]:
clean_digit = knc_clf.predict([Y_test_mod[3600]])
# In[127]:
plt.imshow(clean_digit.reshape(28,28), cmap = matplotlib.cm.binary, interpolation = 'nearest')
# In[128]:
plt.imshow(X_test_mod[3600].reshape(28,28), cmap = matplotlib.cm.binary, interpolation = 'nearest')
# In[129]:
plt.imshow(Y_test_mod[3600].reshape(28,28), cmap = matplotlib.cm.binary, interpolation = 'nearest')
# In[130]:
from sklearn.model_selection import GridSearchCV
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
