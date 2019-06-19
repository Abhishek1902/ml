import pandas as pd


df = pd.read_csv(r"C:\Users\dbda\Desktop\machine learning\ML Test\segmentation.csv",sep=";",index_col=0)
df=df.replace(",",".",regex=True)
#df_dum=pd.get_dummies(df,drop_first=True)
X = df.iloc[:,1:]
dum_x=pd.get_dummies(X, drop_first=True)
y=df.iloc[:,0]
dum_y=pd.get_dummies(y, drop_first=True)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(dum_x, dum_y, test_size = 0.3, 
                                                    random_state=42)
svm=SVC(kernel='linear')
svm1=svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

## Compute predicted probabilities: y_pred_prob
#y_pred_prob = svm.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred)

