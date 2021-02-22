import graphviz as graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Criar o modelo multivariado
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import seaborn as sns

df = pd.read_csv('mammographic_masses.data.txt',
                 names=['BI-RADS', 'Idade', 'Forma', 'Margem', 'Densidade', 'Gravidade'])

# Substituir '?' por uma string vazia
df = df.replace('?', None)
# Substituir as strings vazias por NaN
df.replace({"": np.nan}, inplace=True)
# Substituir os valore NaN pela mediana
df = df.fillna(df.median().iloc[0])
# Converter os valores do dataset para int32 para serem interpretados pelas categorias de dados
df = df.astype('int32')
# nova variavel para definir os dados categoricos
newDf = df
newDf = pd.DataFrame()
newDf['Idade'] = df['Idade']
newDf['Forma'] = df['Forma'].astype('category').map({1.0: 'redonda', 2.0: 'oval', 3.0: 'lobular', 4.0: 'irregular'})
newDf['Margem'] = df['Margem'].astype('category').map(
    {1.0: 'circunscrito', 2.0: 'micro-lobulado', 3.0: 'obscurecido', 4.0: 'mal-definido', 5.0: 'espiculado'})
newDf['BI-RADS'] = df['BI-RADS'].astype('category')
newDf['Densidade'] = df['Densidade'].astype('category').map(
    {1.0: 'alto', 2.0: 'medio', 3.0: 'baixo', 4.0: 'contem-gordura'})
newDf['Gravidade'] = df['Gravidade'].astype('category').map({1.0: 'maligno', 0.0: 'benigno'})
newDf.head()

# Como BI-RADS corresponde à categoria do exame, não é um atributo preditivo e será descartado
df = df.drop(['BI-RADS'], axis=1)

# Gráficos gerados para comparação dos dados

# plot = sns.countplot(x='Forma', hue='Gravidade', data=newDf)
# plot.figure.savefig("Forma_countplot.png")
# plot = sns.countplot(x='Densidade', hue='Gravidade', data=newDf)
# plot.figure.savefig("densidade_gravidade.png")
# plot1 = sns.countplot(x='Densidade', data=newDf)
# plot1.figure.savefig("densidade.png")
# plot2 = sns.countplot(x='Margem', hue='Gravidade', data=newDf)
# plot2.figure.savefig("Margem_gravidade.png")
# plot3 = sns.boxplot(x='Gravidade', y='Idade', data=newDf)
# plot3.figure.savefig("Gravidade_Idade.png")
# plt.figure(figsize=(20,10))
# plot4 =sns.heatmap(df.corr(), annot=True, fmt='.0%')
# plot4.figure.savefig("Heatmap.png")
#
# Criar um array com os titulos dos atributos
feature_labels = ['Idade', 'Forma', 'Margem', 'Densidade']
features = df[feature_labels].values
classes = df['Gravidade'].values

# Como alguns modelos requerem a informação normalizadas precisamos de normalizar a informação
normaliser = preprocessing.StandardScaler()
scaled_features = normaliser.fit_transform(features)

### Utilização do One Hot Encoding apenas à forma e margem

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

dfOHE.head(100).to_string("hotEncoding.txt", encoding='utf-8')
df = newDf

# DIVIDIR OS DADOS EM TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(scaled_features, classes, test_size=0.30, random_state=1)

lm = sm.OLS(y_train, X_train).fit()
print(lm.summary(()))

# 1. CLASSIFICAÇÃO COM DECISION TREES
dec_tree = DecisionTreeClassifier(random_state=1)
dec_tree.fit(X_train, y_train)
dot_data = tree.export_graphviz(dec_tree, out_file=None, feature_names=feature_labels)
graph = graphviz.Source(dot_data)

# graph.render("tree",view = True)  # display the decision tree as tree.pdf
decision_tree_score = dec_tree.score(X_test, y_test) * 100

# 2. CLASSIFICAÇÃO COM RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=10, random_state=1)
cv_score = cross_val_score(random_forest, scaled_features, classes, cv=10)
random_forest_score = cv_score.mean() * 100

# 3. CLASSIFICAÇÃO COM SUPPORT VECTOR MACHINE (SVM)
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

# 4. CLASSIFICAÇÃO COM K NEAREST NEIGHBOURS
# Looping from K = 1 to 30, returning the best val
all_knn = []
for i in range(1, 30):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn_cv_score = cross_val_score(knn, scaled_features, classes, cv=10)
    all_knn.append(knn_cv_score.mean())
knn_score = max(all_knn) * 100
k = all_knn.index(knn_score / 100) + 1

# 5. CLASSIFICAÇÃO COM NAIVE BAYES
nb_normaliser = preprocessing.MinMaxScaler()
nb_features = nb_normaliser.fit_transform(features)
nb = MultinomialNB()
nb_cv_score = cross_val_score(nb, nb_features, classes, cv=10)
nb_score = nb_cv_score.mean() * 100

# 6. CLASSIFICAÇÃO COM A REGRESSÃO LOGISTICA
log_reg = LogisticRegression()
log_reg_cv_score = cross_val_score(log_reg, scaled_features, classes, cv=10)
log_reg_score = log_reg_cv_score.mean() * 100


# 7. CLASSIFICAÇÃO COM A NEURAL NETWORK
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))
    # Técnica de regularização para reduzir o overfitting
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # teste com rmsprop
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Teste com Overfitting
    # reg_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0)

   # Ao utilizar o early stopping tem piores resultados

   # reg_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100,
   # callbacks=EarlyStopping(monitor='val_loss'), batch_size=10, verbose=2)
    return model

def modelAccuracy():
    model = create_model()
    reg_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=2)


    # Avaliação do modelo
    _, accuracy = model.evaluate(X_train, y_train, verbose=0)
    plt.plot(reg_history.history['loss'], label='train')
    plt.plot(reg_history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    print('Accuracy: %.2f' % (accuracy * 100))
    return accuracy * 100


neural = create_model()
print("Resultados da classificação:\n")
print(f'Support Vector Machine linear kernel = {svc_linear_score} %')
print(f'Support Vector Machine polynomial kernel = {svc_polynomial_score} %')
print(f'Support Vector Machine rbf kernel = {svc_rbf_score} %')
print(f'K Nearest Neighbours= {knn_score} % for k = {k}')
print(f'Decision Trees = {decision_tree_score} %')
print(f'Random Forest = {random_forest_score} %')
print(f'Naive Bayes = {nb_score} %')
print(f'Logistic Regression = {log_reg_score} %')
print(f'Neural Network = {neural} %')
