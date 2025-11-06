import streamlit as st
import pandas as pd

st.title("Teste Streamlit")
st.write("Se você está vendo isso, o Streamlit está funcionando!")

# Teste de carregamento de CSV
import os
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
st.write(f"Arquivos CSV encontrados: {len(csv_files)}")

