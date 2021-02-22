import pandas as pd
import numpy
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import preprocessing,metrics
from sklearn.model_selection import cross_val_score, train_test_split

# from sklearn.cross_validation import train_test_split

# Read from CSV file
mammograph_of_data = pd.read_csv('mammographic_masses.data.txt',na_values=['?'],names=['BI-RADS','age','shape','margin', 'density','severity'])
mammograph_of_data.dropna(inplace=True)

# Define my_features and label
my_features = mammograph_of_data[['age','shape','margin','density']].values
my_class = mammograph_of_data[['severity']].values

# We are taking 4 my_features as deterministic my_features
name_features = ['age','shape','margin','density']

# Noramalizing our data using preprocessing
normalized = preprocessing.StandardScaler()
normalized_Features = normalized.fit_transform(my_features)
print(normalized_Features)

# Split train and test data using train_test_split
X_train,X_test,Y_train, Y_test =train_test_split(normalized_Features,my_class,test_size=0.2,random_state=1)
#Naive Bayes Model
My_Model = GaussianNB()
My_Model.fit(X_train,Y_train.ravel())
y_pred = My_Model.predict(X_test)

#K-fold cross validation
#model_scores = cross_val_score(My_Model,normalized_Features,my_class.ravel(), cv= 10)
#print('Accuracy of our model using 10-fold validation is: ',model_scores.mean())

# Taking output from user
age = int(input('Enter age: '))
shape = int(input('Enter shape [Value 1 to 5]: '))
margin = int(input('Enter margin [Value 1 to 5]: '))
density = int(input('Enter density [Value 1 to 5]: '))
# Output of the model which predicts whether its benign and malignant
features_new = [[age,shape,margin,density]]
print(features_new)
features_norm_test = normalized.fit_transform(features_new)
y_pred_val = My_Model.predict(features_norm_test)
print("y" , y_pred_val)

for i in y_pred_val:
    if i == 0:
        print('The predicted result from the My_Model is: Benign')
    else:
        print('The predicted result from the My_Model is: Malignant')