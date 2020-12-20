import numpy as np
import pickle
import pandas as pd
import os
import joblib
import streamlit as st
import seaborn as sns

from PIL import Image


@st.cache
def load_dataset(dataset):
    df = pd.read_csv(dataset)
    return df

class_label = {'SETOSA':0, "VERSICOLOR":1, 'VERGINICA':2}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model



def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Steamlit IRIS Flower Classification App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    menu = ["EDA", "Prediction", "About"]

    choices = st.sidebar.selectbox("Select Activities", menu)

    if(choices == 'EDA'):
        st.header("Data Analysis")

        data = load_dataset('iris.csv')
        
        if st.checkbox("Show Head"):
            st.dataframe(data.head())

        if st.checkbox("Show Tail"):
            st.dataframe(data.tail())

        if st.checkbox("Show Summary"):
            st.write(data.describe())

        if st.checkbox("Show Shape"):
            st.write(data.shape)

        if st.checkbox("Species Count"):
            st.write(data['Species'].value_counts())

        st.header('Data Visualization')

        if st.checkbox("Plot Species"):
            sns.countplot(data['Species'])
            st.pyplot()

        st.subheader("Histogram")

        if st.checkbox('Sepal Length'):
            sns.distplot(data['SepalLengthCm'], kde=False, color='red')
            st.pyplot()

        if st.checkbox('Sepal Width'):
            sns.distplot(data['SepalWidthCm'], kde=False, color='red')
            st.pyplot()

        if st.checkbox('Petal Length'):
            sns.distplot(data['PetalLengthCm'], kde=False, color='red')
            st.pyplot()

        if st.checkbox('Petal Width'):
            sns.distplot(data['PetalWidthCm'], kde=False, color='red')
            st.pyplot()

        st.subheader("Box Plot")

        if st.checkbox('Sepal Length and Species'):
            sns.boxplot(x = 'Species', y = 'SepalLengthCm', data = data)
            st.pyplot()

        if st.checkbox('Sepal Width and Species'):
            sns.boxplot(x = 'Species', y = 'SepalWidthCm', data = data)
            st.pyplot()

        if st.checkbox('Petal Length and Species'):
            sns.boxplot(x = 'Species', y = 'PetalLengthCm', data = data)
            st.pyplot()

        if st.checkbox('Petal Width and Species'):
            sns.boxplot(x = 'Species', y = 'PetalWidthCm', data = data)
            st.pyplot()

        st.subheader("Scatter Plot")

        if st.checkbox('Sepal Length and Sepal Width'):
            sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', hue='Species', data = data)
            st.pyplot()

        if st.checkbox('Petal Length and Petal Width'):
            sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', hue='Species', data = data)
            st.pyplot()

        st.subheader("Pair Plot")

        if st.checkbox("Pair Plot"):
            data = data.drop(['Id'], axis=1)
            sns.pairplot(data, hue='Species')
            st.pyplot()

        st.subheader("Correlation Matrix")

        if st.checkbox("Correlation Matrix"):
            matrix = data.iloc[:, 1:5].corr()
            sns.heatmap(matrix, annot = True)
            st.pyplot()
            st.markdown("**Observation: Petal Length and Petal Width are Highly correlated feature**")



    if(choices == 'Prediction'):
        st.subheader('Prediction')

        sepal_length = st.slider("Sepal Length", 0.0, 10.0)
        petal_length = st.slider("Petal Length", 0.0, 10.0)
        sepal_width = st.slider("Sepal Width", 0.0, 10.0)
        petal_width = st.slider("Petal Width", 0.0, 10.0)
        
        st.subheader("Selected Data")
        sample_data = [sepal_length, petal_length, sepal_width, petal_width]
        st.write(sample_data)

        prep_data = [sample_data]

        model_choice = st.selectbox("Select Model", ["knn", "Naive Bayes", "Logistic Regression", "SVM", "Decision Trees"])
        
        if st.button("Classify"):
            if model_choice == "knn":
                predictor = load_prediction_model("clf_1.pkl")
                prediction = predictor.predict(prep_data)

            elif model_choice == "Naive Bayes":
                predictor = load_prediction_model("clf_2.pkl")
                prediction = predictor.predict(prep_data)

            elif model_choice == "Logistic Regression":
                predictor = load_prediction_model("clf_3.pkl")
                prediction = predictor.predict(prep_data)

            elif model_choice == "SVM":
                predictor = load_prediction_model("clf_4.pkl")
                prediction = predictor.predict(prep_data)

            elif model_choice == "Decision Trees":
                predictor = load_prediction_model("clf_5.pkl")
                prediction = predictor.predict(prep_data)

            final_result = get_key(prediction, class_label)
            st.success(final_result)

    

    if(choices == 'About'):
        st.subheader('About')
        st.text("IRIS Flower Classification")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    