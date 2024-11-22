import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tempfile
from keras import *
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import axes3d
from streamlit_option_menu import option_menu
import tempfile
import os

import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEX,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO usertable(username,password) VALUES (?,?)', (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM usertable WHERE username = ? AND password = ?', (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM usertable')
    data = c.fetchall()
    return data


# efacer menu
hide_st_style = """
              <style>
              #MainMenu {visibility: hidden;}
              footer {visibility: hidden;}
              header {visibility: hidden;}
              <style>
              """
st.markdown(hide_st_style, unsafe_allow_html=True)


def login():
    st.title("AUTHENTIFICATION")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        create_usertable()
        result = login_user(username, password)
        if result:
            st.session_state.logged_in = True
            st.success("Login Correct")
        else:
            st.error("Mots de passe ou usename incorrect")


def Acueille():
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Contexte du projet", "Exploration des données", "Analyse de données",
                     "Modelisation et prediction", "Utilisateur", "Densimètre"],
            icons=["book", "database-fill-down", "bar-chart-fill", "repeat", "people-fill", "speedometer2"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "20px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"}
            },
        )
        if st.button("Quiter"):
            st.session_state.logged_in = False
            st.experimental_rerun()

    if selected == "Contexte du projet":
        st.title(f"{selected}")
        st.write(
            "Ce projet s'inscrit dans le contexte Laboratoire. L'objectif est de faire une prediction de résultat d'un taux de chlore de l'eau prise dans un tybe a essai à partir d'une image prise par la camera du télephone")
        st.write(
            "Nous avons à notre disposition un fichier rvb.csv, qui contient les paramètres rvb des couleurs de la taux de chlore que nous avons élaboré. Chaque observation en ligne correspond à un valeur de taux de chlore. Chaque variable en colonnes est une caractéristique de couleurs par paramètres RVB.")
        st.write(
            "Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des information selon certaine axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire le taux de chlore.")
        st.image(["couleur/test1.png", "couleur/test2.png"], width=350)
        st.image(["couleur/chlore1.png", "couleur/chlore2.png"], width=350)


    elif selected == "Exploration des données":
        st.title(f"{selected}")
        iris_data = pd.read_csv('rvb2.csv')
        st.dataframe(iris_data.head())
        st.write("Dimension de notre dataframe")
        st.write(iris_data.shape)



    elif selected == "Analyse de données":
        st.title(f"{selected}")
        iris_data = pd.read_csv('rvb2.csv')
        fig = sns.pairplot(iris_data, hue='Valeur')
        st.pyplot(fig)



    elif selected == "Modelisation et prediction":
        st.title(f"{selected}")
        iris_data = pd.read_csv('rvb2.csv')

        # Vérification des données
        st.write("Vérification des premières lignes du dataset")
        st.dataframe(iris_data.head())

        st.write("Dimension du dataset")
        st.write(iris_data.shape)

        # Encodage de la colonne cible
        label_encoder = LabelEncoder()
        iris_data['Valeur'] = label_encoder.fit_transform(iris_data['Valeur'])

        # Conversion en numpy array
        np_iris = iris_data.to_numpy()

        # Séparation des features et labels
        x_data = np_iris[:, 0:3]  # Les colonnes RVB
        y_data = np_iris[:, 3]  # La colonne cible

        # Normalisation des features
        scaler = StandardScaler().fit(x_data)
        x_data = scaler.transform(x_data)

        # Vérification de one-hot encoding
        NB_CLASSES = len(np.unique(y_data))
        y_data = tf.keras.utils.to_categorical(y_data, NB_CLASSES)

        # Vérification des données transformées
        st.write("Données d'entraînement après transformation")
        st.write(x_data.shape, y_data.shape)

        # Division en données d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=42)
        st.write("Données d'entraînement et de test")
        st.write(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Définition du modèle de réseau de neurones
        st.write("## Entrainement du modèle")
        if st.checkbox("Lancer l'entrainement du modèle"):
            from tensorflow import keras

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(128, input_shape=(3,), activation='relu', name='Hidden-layer-1'))
            model.add(keras.layers.Dense(128, activation='relu', name='Hidden-layer-2'))
            model.add(keras.layers.Dense(NB_CLASSES, activation='softmax', name='Output-layer'))

            # Compilation du modèle
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

            # Callback personnalisé pour la progression
            progress_bar = st.progress(0)

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def __init__(self, epochs):
                    self.epochs = epochs
                    self.current_epoch = 0

                def on_epoch_end(self, epoch, logs=None):
                    self.current_epoch += 1
                    progress_bar.progress(self.current_epoch / self.epochs)

            # Entraînement du modèle
            VERBOSE = 1
            BATCH_SIZE = 16
            EPOCHS = 105
            VALIDATION_SPLIT = 0.2

            history = model.fit(X_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                verbose=VERBOSE,
                                callbacks=[StreamlitCallback(EPOCHS)],
                                validation_split=VALIDATION_SPLIT)

            # Affichage des résultats
            st.write("Historique de l'entraînement :")
            st.line_chart(pd.DataFrame(history.history)["accuracy"])

            # Évaluation du modèle sur les données de test
            st.write("Évaluation sur les données de test :")
            evaluation = model.evaluate(X_test, y_test)
            st.write(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

            # Sauvegarde du modèle (si besoin)
            # model.save("rvb_model.h5")

            # Prediction
            if st.checkbox("Lancer la prediction"):
                cap = cv2.VideoCapture(0)
                address = "http://192.168.0.101:8080/video"
                st.title("Video Capture")
                frame_placholder = st.empty()
                cap.open(address)
                play_button_press = st.button("Play")
                stop_button_press = st.button("stop")

                # Création fichier temporaire pour les données capturées
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')

                while cap.isOpened() and not stop_button_press:
                    ret, frame = cap.read()

                    if not ret:
                        st.write("La capture a pris fin.")
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, _ = frame.shape
                    cx = int(width / 2)
                    cy = int(height / 2)
                    pixel_center = frame[cy, cx]
                    r, v, b = int(pixel_center[0]), int(pixel_center[1]), int(pixel_center[2])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)
                    frame_placholder.image(frame, channels="RGB")
                    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_press:
                        break
                    print(r, v, b)
                    df = pd.DataFrame([[r, v, b]], columns=['r', 'v', 'b'])
                    df.to_csv('tempFile.csv', index=False)

                # Utilisation du modèle pour la prédiction
                bf = pd.read_csv('tempFile.csv')
                new_r, new_v, new_b = bf.iloc[0]['r'], bf.iloc[0]['v'], bf.iloc[0]['b']
                st.write(new_r, new_v, new_b)

                prediction_input = [[new_r, new_v, new_b]]
                scaled_input = scaler.transform(prediction_input)
                raw_prediction = model.predict(scaled_input)
                prediction = np.argmax(raw_prediction)
                with st.spinner("Prédiction en cours..."):
                    import time
                    time.sleep(5)
                st.write("La prédiction est :", label_encoder.inverse_transform([prediction]))


    elif selected == "Utilisateur":
        st.title(f"{selected}")
        st.subheader("Profiles d'utilisateurs")
        user_result = view_all_users()
        clean_db = pd.DataFrame(user_result, columns=["Username", "Password"])
        st.dataframe(clean_db)

        option = st.selectbox("Selectionner l'action",
                              ("", "Nouveau Compte", "Modifier un compte", "Supprimer un compte"))

        if option == "Nouveau Compte":
            st.subheader("Créer un nouveau compte")
            new_user = st.text_input("Username")
            new_password = st.text_input("Mot de passe", type='password')

            if st.button("Crée"):
                add_userdata(new_user, new_password)
                st.success("Nouveau profils enregistrer")
                st.info("Aller au Menu Login")

        elif option == "Modifier un compte":
            st.subheader("Modifier un compte")
            up_username = st.text_input("Nameuser")
            up_user = st.text_input("Username")
            up_password = st.text_input("Mots de passe", type='password')

            if st.button("Modifie"):
                req = "update usertable set username = ? , password = ? where username = ?"
                val = (up_user, up_password, up_username)
                c.execute(req, val)
                conn.commit()
                st.success("modification enregistrer")


        elif option == "Supprimer un compte":
            st.subheader("Supprimer un compte")
            sup_user = st.text_input("Username")
            if st.button("Supprimer"):
                req = "delete from usertable where username = ?"
                val = (sup_user,)
                c.execute(req, val)
                conn.commit()
                st.success("Supression success")



    elif selected == "Densimètre":
        st.title(f"{selected}")
        st.subheader("Prédiction de chlore par densité")
        if st.checkbox("Entrer la valeur de la densité"):
            densite = st.number_input("Entre la valeur de la densité:")

        if st.button('predict'):
            # define model architecture
            model = Sequential()
            model.add(layers.Dense(units=3, input_shape=[1]))
            model.add(layers.Dense(units=64))
            model.add(layers.Dense(units=1))
            # compile the model
            model.compile(optimizer='rmsprop', loss='mean_squared_error')
            # define training data

            xs = np.array(
                [1.007, 1.014, 1.021, 1.028, 1.036, 1.044, 1.051, 1.059, 1.067, 1.075, 1.083, 1.091, 1.099, 1.108,
                 1.116, 1.125, 1.134], dtype=float)
            ys = np.array(
                [2.8, 5.5, 8.0, 10.5, 13.5, 16.0, 18.5, 21.0, 23.0, 25.0, 27.5, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
                dtype=float)

            # Callback personnalisé pour afficher la progression
            progress_bar = st.progress(0)
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def __init__(self, epochs):
                    self.epochs = epochs
                    self.current_epoch = 0

                def on_epoch_end(self, epoch, logs=None):
                    self.current_epoch += 1
                    progress_bar.progress(self.current_epoch / self.epochs)

            # train the model
            epochs = 4000
            # ketriaka = model.fit(xs, ys, epochs=2100, allbacks=[StreamlitCallback(epochs)])

            ketriaka = model.fit(xs, ys, epochs=epochs, callbacks=[StreamlitCallback(epochs)], verbose=0)

            score = model.evaluate(xs, ys, verbose=0)
            import matplotlib.pyplot as plt
            st.set_option('deprecation.showPyplotGlobalUse', False)
            pd.DataFrame(ketriaka.history)["loss"].plot(figsize=(8, 5))

            st.pyplot(plt.show())
            st.write(score)
            # make prediction
            st.write("Valeur de chlore est de : ")
            st.write(model.predict(np.array([densite], dtype=float)))


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    Acueille()
else:
    login()


