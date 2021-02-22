from datetime import date, datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from dateutil.relativedelta import relativedelta

# y_pred = My_Model.predict(X_test)


df = pd.read_csv('mammographic_masses.data.txt',
                 names=['BI-RADS', 'Idade', 'Forma', 'Margem', 'Densidade', 'Gravidade'])
# Replace '?' with Empty String
df = df.replace('?', None)
# Replace empty string with NaN
df.replace({"": np.nan}, inplace=True)

# Replace NaN with row median
df = df.fillna(df.median().iloc[0])
# Drop BI-RADS row because it is not a predictive attribute
df = df.drop(['BI-RADS'], axis=1)

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
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=2)

My_Model = model
My_Model.fit(X_train,y_train.ravel())
y_pred = My_Model.predict(X_test)




def prediction(birads, date, shape, margin, density):
    # Output of the model which predicts whether its benign and malignant
    age = getAge(date)
    features_new = [[age, shape, margin, density]]
    normalized = preprocessing.StandardScaler()
    features_norm_test = normalized.fit_transform(features_new)

    y_pred_val = My_Model.predict_classes(normaliser.transform(features_new))
    print(y_pred_val)
    for i in y_pred_val:
        if i == 0:
            return 'Benigno'
        else:
            return 'Maligno'



def getAge(birth):
    today = date.today()

    date_of_birth = datetime.strptime(birth, "%d/%m/%Y")
    print(date_of_birth)
    birthDate = date_of_birth
    age = today.year - birthDate.year - ((today.month, today.day) <
                                         (birthDate.month, birthDate.day))
    return age
