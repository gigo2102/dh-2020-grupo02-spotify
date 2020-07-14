#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump, load
import requests

def give_rec(title, sig, indices, df):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    tema_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return df['tema'].iloc[tema_indices]


# descarga de archivos con streaming desde google drive
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def get(id, destination):
	URL = "https://docs.google.com/uc?export=download"
	session = requests.Session()
	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)
	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)
	save_response_content(response, destination)
	return destination

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768
	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)


# descargamos del google drive los archivos
# leemos los archivos segun documentacion de sklearn con joblib
@st.cache
def get_idx():
	return get('13LBncsc9sU4ATRqh0civwYV0fpf2Uckv', 'idx.joblib')
	
@st.cache
def get_df():
	return get('1Kls7WC-KpLMblIDoptn2OBR-6QYEfDTb', 'df.joblib')

@st.cache
def get_model():
	return get('1zob_EMMBhBAj-hNDDTb7jOon9pQr2sDj', 'model.joblib')

indices2 = load(get_idx())
df2 = load(get_df())
sig2 = load(get_model())

# armamos la pagina
st.title('Te recomendamos canciones!')

canciones_list = st.sidebar.selectbox("Elegir Tema para recomendaciones por letra", df2["tema"].unique())
q = df2.query(f"tema=='{canciones_list}'")# .iloc[0]
cancion_elegida = q.iloc[0]['tema']
st.write(f'Elegiste la cancion: {cancion_elegida} - Tus recomendaciones por letra son:')
# escribimos recomendaciones
results = give_rec(cancion_elegida, sig2, indices2, df2)
st.write(results)

# =====================================================

def give_rec_audio(title, indices, df, model_nn, X_sc):
	# Indice de la cancion
	idx = indices[title]
	# indices mas cercanos a ese indice 
	vecinos = model_nn.kneighbors(X_sc[idx,:].reshape(1,-1), n_neighbors=6, return_distance=False)
	# Top de canciones mas similares
	temas = dict({})
	for i in vecinos:
		for recomendados in list(df['tema'][i]):
			temas[str(recomendados).lower()] = recomendados
	return pd.DataFrame(list(temas.values()))
    
indices_audio = load('idx_audio.joblib')
df_audio = load('df_audio.joblib')
model_audio = load('model_audio.joblib')
X_sc_audio = load('X_sc_audio.joblib')

canciones_list_audio = st.sidebar.selectbox("Elegir Tema para recomendaciones por audio", df_audio["tema"].unique()[:1000])
q_audio = df_audio.query(f"tema=='{canciones_list_audio}'")# .iloc[0]
cancion_elegida_audio = q_audio.iloc[0]['tema']
st.write(f'Elegiste la cancion: {cancion_elegida_audio} - Tus recomendaciones por audio son:')
# escribimos recomendaciones
results_audio = give_rec_audio(cancion_elegida_audio, indices_audio, df_audio, model_audio, X_sc_audio)
st.write(results_audio)


# In[ ]:




