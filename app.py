import streamlit as st
import spacy
nlp = spacy.load('en_core_web_sm')
import tfidf
from spacy import displacy
import convertapi
import string
import re
import judgement

convertapi.api_secret = 'cA4oXDkB80RGJE8c'

st.title('Legal Document Brief Generator')

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            st.write(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        st.write('No named entities found.')


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    result = convertapi.convert('txt', {'File': 'uploaded_file.pdf'}, from_format='pdf')
    
    result.file.save('converted.txt')
    with open('converted.txt', 'r') as f:
        original_text = f.read()
    summary = tfidf.summarize(original_text)
    summary = re.sub(r'\W+', ' ', summary)
    summary = ''.join(c for c in summary if c in string.printable)
	#convert all sentences to single string

    # for i in nlp(summary).ents:
    #     if(i.label_ == 'CARDINAL'):
    #         continue
    #     st.write(i.text,":", i.label_)    

    st.write(displacy.render(nlp(summary), style='ent', jupyter=False,options={'ents':['DATE','PERSON','MONEY','ORG','GPE','NORP','QUANTITY','LOC','LAW','MONEY']}), unsafe_allow_html=True)



st.markdown('# Judgement Prediction')

text_input = st.text_input("Enter Facts")
if text_input:
    predict = judgement.predict(text_input)
    if(predict[0][0] > 0.5):
        st.write(f"The judge will rule in the first party's favor {(predict[0][0]*100).round(2)}% of the time")
    else:
        st.write(f"The judge will rule in the second party's favor {((1-predict[0][0])*100).round(2)}% of the time")