import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD

# Load the data
us_canada_user_rating_pivot2 = pd.read_csv("C:\\Users\\91866\\Phyton\\Project_Recommendation\\filename.csv", encoding='latin1')
X = us_canada_user_rating_pivot2.values.T

# Load the model
SVD = pickle.load(open('C:\\Users\\91866\\Phyton\\Project_Recommendation\\ttrained_model.sav', 'rb'))

# Compute the correlation matrix
SVD = TruncatedSVD(n_components=12)
matrix = SVD.fit_transform(X)
corr = np.corrcoef(matrix)

# Get the book titles
us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)

# Function to get similar books
def similar_books(book_name):
    coffey_hands = us_canada_book_list.index(book_name)
    corr_coffey_hands  = corr[coffey_hands]
    l = list(us_canada_book_title[(corr_coffey_hands<1.0) & (corr_coffey_hands>0.95)])
    return l

# Streamlit app
st.title('Book Recommendation System')
book_name = st.text_input("Enter the name of a book from the list:")
if st.button('Recommend'):
    result = similar_books(book_name)
    st.write(result)
