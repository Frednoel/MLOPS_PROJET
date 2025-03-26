import streamlit as st
import pandas as pd 
import numpy as np 
import gzip
import json
import pickle
import re
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import arff
import pandas as pd
import os




st.title(" VISUALISATION DATA MANAGEMENT")
st.sidebar.title("MENU DEROULANT")

st.markdown("<u>Notre application de visualisation des analyses effectuées lors du data management</u>", unsafe_allow_html=True)



def wrangle(filepath):
    with open(filepath, 'r') as file:
        data = arff.load(file)
    
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    
    df.columns = [re.sub(pattern='Attr',repl='feat_',string=col) for col in df.columns]
    df.rename(columns={'class':'bankrupt'},inplace=True)
    
    # Change dtype of the Labels columns
    df['bankrupt'] = df['bankrupt'].astype(np.int64)
    df['status'] = df['bankrupt'].apply(lambda x: "The company in bankrupt" if x == 1 else "The company is safe")

    # column is the most missing value
    df.drop(columns='feat_37',inplace=True)
    return df

url = "data/poland.arff"

df = wrangle(url)


print(df.head())

st.dataframe(df)
st.sidebar.subheader("Bankrupt classes 0 or 1")
bank_rupt = st.sidebar.radio('Type de classes',('0','1'))


st.sidebar.markdown(df.query('bankrupt==@bank_rupt')[["status"]])
                    
st.sidebar.markdown("### Targets Bankrupt Frequences")
select = st.sidebar.selectbox('Visualisation type', ['histogram', 'Pie chart'], key='1')
val_count = df["bankrupt"].value_counts(normalize = True)
st.write(val_count)
val_count = pd.DataFrame({'type classes': val_count.index, 'numbers' : val_count.values})


if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Numbers of classes bankrupt")
    if select == "histogram":
        fig = px.bar(val_count, x = 'type classes',y='numbers', color='numbers', height =500)
        st.plotly_chart(fig)
    else:
        fig=px.pie(val_count, values='numbers', names='type classes')
        st.plotly_chart(fig)
        


# Summary statistics for `feat_27`
st.sidebar.markdown("### Distribution  Profit/Expenses Ratio (feat_27)")
select = st.sidebar.selectbox('Visualisation type', ['histogram', 'Box plot'])
st.write("Distribution  Profit/Expenses Ratio, by Bankruptcy status")


# Créer un graphique en barres avec Plotly Express
if not st.sidebar.checkbox("caché", True):
    st.markdown("### Numbers of classes bankrupt")
    if select == "histogram":
        fig = px.bar(
            df, 
            x='bankrupt', 
            y='feat_27', 
            color='feat_27', 
            labels={'bankrupt': 'Bankrupt Class', 'feat_27': 'POA / Financial Expenses'},
            height=500, 
            title="Distribution of Profit/Expenses Ratio by Bankruptcy Class"
        )
        st.plotly_chart(fig)
    else:
        fig = px.box(
            df, 
            x='bankrupt', 
            y='feat_27', 
            color='bankrupt', 
            labels={'bankrupt': 'Bankrupt Class', 'feat_27': 'POA / Financial Expenses'},
            height=500, 
            title="Box Plot of Profit/Expenses Ratio by Bankruptcy Class"
        )
        st.plotly_chart(fig)
        
    

        
st.write("On remarque en remarquant le box blot qu'une bonne partie des données semble se concentrée dans un intervalle precis. Les ecarts de valeurs nous montre qu'il y a enormement de valeurs aberrantes.")
q1,q9= df["feat_27"].quantile([0.1,0.9])
mask = df["feat_27"].between(q1,q9)  

fig = px.box(
            df[mask], 
            x='bankrupt', 
            y='feat_27', 
            color='bankrupt', 
            labels={'bankrupt': 'Bankrupt Class', 'feat_27': 'POA / Financial Expenses'},
            height=500, 
            title="Box Plot of Profit/Expenses Ratio by Bankruptcy Class")

st.plotly_chart(fig)

st.write("MATRICE DE CORRELATION")

# Calcul de la matrice de corrélation en retirant la colonne 'bankrupt'
corr = df.drop(columns=["bankrupt","status"]).corr()

# Création de la heatmap avec Plotly Express
fig = px.imshow(
    corr,
    text_auto=True,               # Affiche les valeurs de corrélation sur la heatmap
    aspect="auto",                # Ajuste l'aspect de la figure
    color_continuous_scale='RdBu_r',# Choix de la palette de couleurs
    title="Matrice de Corrélation"
)

st.plotly_chart(fig)



st.write("## MATRICE DE CONFUSION MODELE ARBRE DE DECISION")

target = "bankrupt"
X = df.select_dtypes(include= "float64")
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_reg= make_pipeline(
    SimpleImputer(strategy="median"),DecisionTreeClassifier(random_state=42)
                  
)


st.write("MATRICE DE CONFUSION MODELE RANDOM FOREST ET GRADIENT BOOSTING")

 # Pipeline du model


clf = make_pipeline(
    SimpleImputer(),RandomForestClassifier(random_state=42)
)
clf

cv_acc_scores = cross_val_score(clf , X_train, y_train, cv=5, n_jobs=-1)
print(cv_acc_scores)

params = {
    "simpleimputer__strategy":["mean","median"],
    "randomforestclassifier__n_estimators" : range(25,100,25),
    "randomforestclassifier__max_depth" : range(10,50,10)
}
params

model = GridSearchCV(
     clf,
     param_grid=params,
     cv=5,
     n_jobs=-1,
     verbose=1
)
print(model)

# Train model
model.fit( X_train, y_train)

st.write("RESULT MODELE RANDOM FOREST ")

cv_results =pd.DataFrame(model.cv_results_)
st.dataframe(cv_results.head(10))

st.markdown("** En plus des scores de précision pour tous les différents modèles que nous avons essayés lors de notre recherche dans la grille, nous pouvons voir combien de temps il a fallu à chaque modèle pour s’entraîner. Examinons de plus près comment les différents paramètres d’hyperparamètres affectent le temps d’entraînement.")
# Extract best hyperparameters


# Get feature names from training data
features = X_train.columns
# Extract importances from model
importances = model.best_estimator_.named_steps["randomforestclassifier"].feature_importances_
# Create a series with feature names and importances
feat_imp = pd.Series(importances, index=features)
# Plot 10 most important features
feat_imp.tail(10).plot(kind="barh")


# Tracer des elements 


# Afficher les meilleurs paramètres
st.subheader("Meilleurs paramètres du modèle")
st.write(model.best_params_)

# Afficher le rapport de classification
st.subheader("Rapport de classification")
y_pred = model.predict(X_test)
clf_report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(clf_report).transpose())

# Afficher la matrice de confusion
st.subheader("Matrice de confusion")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Prédit')
ax.set_ylabel('Réel')
st.pyplot(fig)

# Afficher l'importance des features
st.subheader("Importance des features (Top 10)")
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

fig, ax = plt.subplots()
feat_imp.head(10).plot(kind='barh', ax=ax)
ax.set_xlabel("Importance Gini")
ax.set_ylabel("Feature")
st.pyplot(fig)

 
 # MODELE GRADIENTBOOSTING