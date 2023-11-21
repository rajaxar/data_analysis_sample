import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space 
from statsmodels.formula.api import ols
import streamlit as st
import altair as alt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Coding Practice Submission")
st.write("""
        This app uses Streamlit to visualize the monthly
        participation rates in the DPS school lunch program.
         """)

lunch_file = st.file_uploader("Upload a CSV with school lunch data (default provided.)", type="csv")
st.write("Make sure your data has columns for participation rates in months 1-13.")

@st.cache_data()
def load_file(f):
    if f:
        df = pd.read_csv(f)
        if 'Month 13' not in df.columns:
            st.error("Please upload a CSV with a column for month 13 participation rates. Reverting to default data.")
            df = pd.read_csv('lunch.csv')
    else:
        df = pd.read_csv('lunch.csv')
    return df

lunch_df = load_file(lunch_file)
lunch_df.rename(columns={'Month 01': 'Month1',
                         'Month 02': 'Month2',
                         'Month 03': 'Month3',
                         'Month 04': 'Month4',
                         'Month 05': 'Month5',
                         'Month 06': 'Month6',
                         'Month 07': 'Month7',
                         'Month 08': 'Month8',
                         'Month 09': 'Month9',
                         'Month 10': 'Month10',
                         'Month 11': 'Month11',
                         'Month 12': 'Month12',
                         'Month 13': 'Month13'
                         }, inplace=True)

st.dataframe(lunch_df.head(7))

st.subheader("Let's Build a Model!")
st.write("We will use data about the participation rates in previous months to predict the participation rate in month 13.")

st.write("What months should we use to predict month 13?")
options = st.multiselect("Select the months to use to predict month 13:", 
                ['Month1', 'Month2', 'Month3', 'Month4', 'Month5', 'Month6', 'Month7', 'Month8', 'Month9', 'Month10', 'Month11', 'Month12'])

if len(options) == 0:
    options = ['Month1', 'Month2', 'Month3', 'Month4', 'Month5', 'Month6', 'Month7', 'Month8', 'Month9', 'Month10', 'Month11', 'Month12']

new_df = lunch_df[options + ['Month13']]
new_df = new_df.dropna()
new_df = new_df.astype('float64')

tab1, tab2, tab3 = st.tabs(["Picking our Months", "Visualizing our Results", "Predicting Month 13"])

with tab1:
    add_vertical_space(2)

    st.write("Here is a correlation matrix of the selected months:")
    col1, col2 = st.columns(2, gap="large")
    corr = new_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr = corr.mask(mask).iloc[1:, :-1]
    
    with col1:
        st.dataframe(corr)
    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=False, ax=ax)
        st.pyplot(fig)

    add_vertical_space(5)

    st.write("Let's build a linear regression model to predict month 13 participation rate based on the selected months.")
    month_vars = 'Month13 ~' + ' + '.join(options)
    model = ols(month_vars, data=new_df).fit()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.write(model.summary().tables[1].as_html(), unsafe_allow_html=True)
    with col2:
        rSquared = model.rsquared
        monthsWithPValues = model.pvalues[model.pvalues < 0.05]
        monthsWithPValues = monthsWithPValues.drop('Intercept')
        st.write("R-squared:", rSquared)
        st.write("Months with `p-values < 0.05`:", monthsWithPValues.index.tolist())
        add_vertical_space(1)
        st.write("Try removing months with high p-values to see how it affects the model!")

with tab2:
    add_vertical_space(2)
    st.write('The computed model is: ')
    latex_equation = ''
    for i in range(len(model.params)):
        if i == 0:
            latex_equation += str(round(model.params.iloc[i], 2))
        else:
            latex_equation += ' + ' + str(round(model.params.iloc[i], 2)) + ' * ' + options[i-1].replace('Month', 'x_{') + '}'
    
    latex_equation = 'x_{13} = ' + latex_equation + ' + \\epsilon'
    st.latex(latex_equation.replace('+ -', '- '))
    add_vertical_space(5)

    st.write("Here is a scatterplot of the predicted vs. actual values:")
    predictions = model.predict(new_df)
    fig = alt.Chart(pd.DataFrame({'Actual': new_df['Month13'], 'Predicted': predictions})).mark_circle().encode(
        x='Actual',
        y='Predicted',
    )
    regression = fig \
        .transform_regression('Actual', 'Predicted') \
        .mark_line()
    
    st.altair_chart(fig + regression, use_container_width=True)

with tab3:
    add_vertical_space(2)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        selections = {}
        for option in options:
            selections[option] = st.slider(f"What participation was there in {option.replace('Month', 'Month ')}?", 0, 100, 50) / 100

    with col2:
        st.write("Here are the selections you made:")
        st.write(selections)
        prediction = model.predict(selections)
        st.metric(f"The predicted participation rate for month 13 is:", f"{round(min(max(prediction[0],0),1) * 100, 2)}%")