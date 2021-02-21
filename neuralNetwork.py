import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import preprocessing, metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

import learning

# Read from CSV file
mammograph_of_data = pd.read_csv('mammographic_masses.data.txt', na_values=['?'],
                                 names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
mammograph_of_data.dropna(inplace=True)

# Define my_features and label
my_features = mammograph_of_data[['age', 'shape', 'margin', 'density']].values
my_class = mammograph_of_data[['severity']].values

# We are taking 4 my_features as deterministic my_features
name_features = ['age', 'shape', 'margin', 'density']

# Noramalizing our data using preprocessing
normalized = preprocessing.StandardScaler()
normalized_Features = normalized.fit_transform(my_features)
print(normalized_Features)

# Split train and test data using train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(normalized_Features, my_class, test_size=0.2, random_state=1)
# Naive Bayes Model
# My_Model = GaussianNB()
# My_Model.fit(X_train,Y_train.ravel())
# y_pred = My_Model.predict(X_test)
df = pd.read_csv('mammographic_masses.data.txt',
                 names=['BI-RADS', 'Idade', 'Forma', 'Margem', 'Densidade', 'Gravidade'])
# Replace '?' with Empty String
df = df.replace('?', None)
# Replace empty string with NaN
df.replace({"": np.nan}, inplace=True)

# Replace NaN with row median
df = df.fillna(df.median().iloc[0])

# Create also an array of the feature name labels
feature_labels = ['Idade', 'Forma', 'Margem', 'Densidade']
features = df[feature_labels].values
classes = df['Gravidade'].values

# Because some models and algorithms need the data normalized we need to normalise the data
normaliser = preprocessing.StandardScaler()
scaled_features = normaliser.fit_transform(features)
# Splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, classes, test_size=0.30, random_state=1)
model = Sequential()
model.add(Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
reg_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=2)

# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train, verbose=0)

print('Accuracy: %.2f' % (accuracy * 100))
...
# make probability predictions with the model
predictions = model.predict(X_train)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(classes[np.argmax(rounded)])



def prediction(age, shape, margin, density):
    # Output of the model which predicts whether its benign and malignant
    features_new = [[age, shape, margin, density]]
    features_norm_test = normalized.fit_transform(features_new)
    y_pred_val = model.predict(features_norm_test)
    for i in y_pred_val:
        if i == 0:
            print('The predicted result is: Benign')
        else:
            print('The predicted result is: Malignant')

print(prediction(23,1,3,4))