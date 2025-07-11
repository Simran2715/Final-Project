import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded")


df=pd.read_csv('Combined_data.csv')
st.title('📊 Student Performance Dashboard')

st.sidebar.title('Filters')

pass_fail= st.sidebar.multiselect('Select Option', df['pass_fail'].unique(), default=df['pass_fail'].unique())
topic_difficulty= st.sidebar.multiselect('Select Option', df['topic_difficulty'].unique(), default=df['topic_difficulty'].unique())
dropout= st.sidebar.multiselect('Select Option', df['dropout'].unique(), default=df['dropout'].unique())
# student = st.sidebar.multiselect('Select Option', df['Student_ID'].unique(), default=df['Student_ID'].unique())

filtered_df = df[(df['pass_fail'].isin(pass_fail)) & (df['topic_difficulty'].isin(topic_difficulty)) & (df['dropout'].isin(dropout))] 

# Metrices
col1, col2, col3= st.columns(3)
col1.metric("Total Failures", f"{df['previous_failures'].sum()}")
col2.metric("Average Attendance Rate", f"{df['attendance_rate'].mean():.0f}")
col3.metric("Records", len(filtered_df))

col1, col2, col3 = st.columns(3)

with col1:
    fig_line = px.line(filtered_df, x='Past_Grades', y='pass_fail', color='pass_fail', title='Pass or Fail')
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    score = filtered_df.groupby('topic_difficulty')['Past_Grades'].sum().reset_index()
    fig_bar = px.bar(score, x='topic_difficulty', y='Past_Grades', title='Topic difficulty by Past Grades')
    st.plotly_chart(fig_bar, use_container_width=True)

with col3:
    counts = filtered_df['pass_fail'].value_counts()
    fig_pie= px.pie(counts,labels=counts.index, values=counts.values,title='Marks Distribution')
    st.plotly_chart(fig_pie, use_container_width=True)


col1, col2 = st.columns(2)

with col1:
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    fig_heat=px.imshow(numeric_df.corr(), title='Correlation Heatmap')
    st.plotly_chart(fig_heat, use_container_width=True)

with col2:
    fig = px.bar(filtered_df, x = "attendance_rate", y = "Past_Grades",
             color = "pass_fail", title = "Long-Form Input")
    st.plotly_chart(fig, use_container_width=True)

col1, col2 =st.columns(2)

with col1:
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 1934,
    title = {'text': "dropout"},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # long_df = px.filtered_df()
    fig = px.bar(filtered_df, x="dropout", y="previous_failures", color="dropout", title="Dropout by previous failures")
    st.plotly_chart(fig, use_container_width=True)





st.subheader("Filtered Data")
st.dataframe(filtered_df)
