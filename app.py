import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

#function
def transform_text(text):
  text=text.lower() #lower case
  text=nltk.word_tokenize(text) #tokenization

  #removing special char(keeping only alphanumeric) ,removing stopwords, removing punctuatuion
  y=[]
  for i in text:
    if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation :
      y.append(i)

  text=y[:]
  y.clear()

  #stemming(always tokenize first)
  for i in text:
    y.append(ps.stem(i))

  #join them as a text (finally they shouldnt be as tokens)
  text= " ".join(y)
  return text

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS spam Classifier")

input_sms = st.text_area("Enter the SMS message")



if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vectorized_sms = tfidf.transform([transformed_sms])
    # 3. predict (ham/spam)
    result = model.predict(vectorized_sms)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

