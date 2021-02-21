import sm as sm
from numpy import nan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder

# build multivariate model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics, preprocessing, neighbors, svm
import re
import statsmodels.api as sm

# Cleaning the data - Remove all ? symbols
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv('mammographic_masses.data.txt',
                 names=['BI-RADS', 'Idade', 'Forma', 'Margem', 'Densidade', 'Gravidade'])

# Replace '?' with Empty String
df = df.replace('?', None)
# Replace empty string with NaN
df.replace({"": np.nan}, inplace=True)

# Replace NaN with row median
df = df.fillna(df.median().iloc[0])
# Convert data to float
s = df.astype('int32')  # .astype('float64')
# print table with count, mean, std, min, max and percentiles
# nova variavel com os dados categoricos
newDf = df

newDf = pd.DataFrame()
newDf['Idade'] = s['Idade']
newDf['Forma'] = s['Forma'].astype('category').map({1.0: 'redonda', 2.0: 'oval', 3.0: 'lobular', 4.0: 'irregular'})
newDf['Margem'] = s['Margem'].astype('category').map(
    {1.0: 'circunscrito', 2.0: 'micro-lobulado', 3.0: 'obscurecido', 4.0: 'mal-definido', 5.0: 'espiculado'})
newDf['BI-RADS'] = s['BI-RADS'].astype('category')
newDf['Densidade'] = s['Densidade'].astype('category').map(
    {1.0: 'alto', 2.0: 'medio', 3.0: 'baixo', 4.0: 'contem-gordura'})
newDf['Gravidade'] = s['Gravidade'].astype('category').map({1.0: 'maligno', 0.0: 'benigno'})
newDf.head()
newDf.head(100).to_string("categorizacao1.txt", encoding='utf-8')
df = df.astype('float64')
print(df.head())

# plot = sns.countplot(x='Forma', hue='Gravidade', data=newDf)
# plot.figure.savefig("Forma_countplot.png")
# Drop BI-RADS row because it is not a predictive attribute
df = df.drop(['BI-RADS'], axis=1)

# Como BI-RADS é a categoria do exame, não é um atributo preditivo e será descartado

# Converting Pandas DataFrame to numpy Arrays that can be used by scikit_learn
# Create an array with the date we are going to work with (idade, forma, margem, densidade) and one with the classes (gravidade)
#
# Create also an array of the feature name labels
feature_labels = ['Idade', 'Forma', 'Margem', 'Densidade']
features = df[feature_labels].values
classes = df['Gravidade'].values

# Because some models and algorithms need the data normalized we need to normalise the data
normaliser = preprocessing.StandardScaler()
scaled_features = normaliser.fit_transform(features)

### Using One Hot Encoding to Handle Categorical data

enc = OneHotEncoder(sparse=False, categories='auto')
shapeFeatureArr = enc.fit_transform(df[['Forma']])
shapeFeatureLabels = ['redonda', 'oval', 'lobular', 'irregular']
shapeFeature = pd.DataFrame(shapeFeatureArr, columns=shapeFeatureLabels)
shapeFeature

marginFeatureArr = enc.fit_transform(df[['Margem']])
marginFeatureLabels = ['circunscrito', 'micro-lobulado', 'obscurecido', 'mal-definido', 'espiculado']
marginFeature = pd.DataFrame(marginFeatureArr, columns=marginFeatureLabels)
marginFeature

dfOHE = pd.concat([df[['Idade']], shapeFeature, marginFeature, df[['Densidade', 'Gravidade']]], axis=1)
print('Nominal features are one hot encoded and ordinal features are left as is.')
print(dfOHE.head())

dfOHE.head(100).to_string("categorizacao.txt", encoding='utf-8')
df = newDf
# Splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, classes, test_size=0.30, random_state=1)

lm = sm.OLS(y_train, X_train).fit()
print(lm.summary(()))

# 1. CLASSIFICATION USING DECISION TREES
dec_tree = DecisionTreeClassifier(random_state=1)
dec_tree.fit(X_train, y_train)
decision_tree_score = dec_tree.score(X_test, y_test) * 100

# 2. CLASSIFICATION USING RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=10, random_state=1)
cv_score = cross_val_score(random_forest, scaled_features, classes, cv=10)
random_forest_score = cv_score.mean() * 100

# 3. CLASSIFICATION USING SUPPORT VECTOR MACHINE (SVM)
C = 1.0

# Linear kernel
svc_linear = svm.SVC(kernel='linear', C=C)
svc_linear_cv_score = cross_val_score(svc_linear, scaled_features, classes, cv=10)
svc_linear_score = svc_linear_cv_score.mean() * 100

# Polynomial kernel
svc_polynomial = svm.SVC(kernel='poly', C=C)
svc_polynomial_cv_score = cross_val_score(svc_polynomial, scaled_features, classes, cv=10)
svc_polynomial_score = svc_polynomial_cv_score.mean() * 100

# RBF kernel
svc_rbf = svm.SVC(kernel='rbf', C=C)
svc_rbf_cv_score = cross_val_score(svc_rbf, scaled_features, classes, cv=10)
svc_rbf_score = svc_rbf_cv_score.mean() * 100

# 4. CLASSIFICATION USING K NEAREST NEIGHBOURS
# Looping from K = 1 to 30, returning the best val
all_knn = []
for i in range(1, 30):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn_cv_score = cross_val_score(knn, scaled_features, classes, cv=10)
    all_knn.append(knn_cv_score.mean())
knn_score = max(all_knn) * 100
k = all_knn.index(knn_score / 100) + 1

# 5. CLASSIFICATION USING NAIVE BAYES
nb_normaliser = preprocessing.MinMaxScaler()
nb_features = nb_normaliser.fit_transform(features)
nb = MultinomialNB()
nb_cv_score = cross_val_score(nb, nb_features, classes, cv=10)
nb_score = nb_cv_score.mean() * 100

# 6. CLASSIFICATION USING LOGISTIC REGRESSION
log_reg = LogisticRegression()
log_reg_cv_score = cross_val_score(log_reg, scaled_features, classes, cv=10)
log_reg_score = log_reg_cv_score.mean() * 100


# 7. CLASSIFICATION USING A NEURAL NETWORK
def create_model():
    model = Sequential()
    # adicionamos 6 dimensoes
    model.add(Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    # configuramos 1 output
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # With Overfitting
    # reg_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0)

   #com o early stopping tem piores resultados
   # reg_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100,
   #                         callbacks=EarlyStopping(monitor='val_loss'), batch_size=10, verbose=2)
    reg_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=2)


    # evaluate the keras model
    _, accuracy = model.evaluate(X_train, y_train, verbose=0)

    plt.plot(reg_history.history['loss'], label='train')
    plt.plot(reg_history.history['val_loss'], label='test')
    plt.show()

    print('Accuracy: %.2f' % (accuracy * 100))
    ...
    # make probability predictions with the model
    predictions = model.predict(X_train)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    return accuracy * 100


# model_estimator = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
# neural_network = cross_val_score(model_estimator, scaled_features, classes, cv=10)
# neural_network_score = neural_network.mean() * 100
neural = create_model()
print("CLASSIFICATION ACCURACY RESULTS:")
print(f'Decision Trees = {decision_tree_score} %')
print(f'Random Forest = {random_forest_score} %')
print(f'SVM linear kernel = {svc_linear_score} %')
print(f'SVM polynomial kernel = {svc_polynomial_score} %')
print(f'SVM rbf kernel = {svc_rbf_score} %')
print(f'KNN = {knn_score} % for k = {k}')
print(f'Naive Bayes = {nb_score} %')
print(f'Logistic Regression = {log_reg_score} %')
# print(f'Neural Network = {neural_network_score} %')

print(f'Neural Network = {neural} %')
