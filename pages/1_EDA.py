import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datasets import load_dataset

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

dataset = load_dataset("banking77")
train_df = pd.DataFrame(dataset['train'])
label_names = dataset['train'].features['label'].names
train_df['intent'] = train_df['label'].map(lambda x: label_names[x])
intent_counts = train_df['intent'].value_counts()

st.markdown("## Intent Variable Distribution")
fig, ax = plt.subplots(figsize = (15,7))
intent_counts.plot(kind='bar',ax = ax)
plt.title('Intent Distribution - Banking77 Train Set')
plt.xlabel('Intent')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
st.pyplot(fig,use_container_width=True)
st.divider()

train_df['text_length'] = train_df['text'].apply(lambda x: len(x.split()))
text_lengths = train_df['text_length']

st.markdown("## Text Length Distribution")
fig, ax = plt.subplots(figsize = (15,7))
text_lengths.hist(ax = ax)
plt.title('Text Length Distribution - Banking77 Train Set')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.tight_layout()
st.pyplot(fig,use_container_width=True)

st.markdown("#### Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Mean",round(train_df['text_length'].mean(),1))
col2.metric("Minimum",train_df['text_length'].min())
col3.metric(" Maximum",train_df['text_length'].max())
st.divider()


st.markdown("## Sample Tickets")
df_sample = train_df[["text", "intent"]].sample(10)
st.markdown(
    df_sample.to_html(index=False),
    unsafe_allow_html=True
)