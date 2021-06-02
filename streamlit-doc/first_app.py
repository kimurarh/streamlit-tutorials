import streamlit as st

import numpy as np
import pandas as pd

""" Add Title """
st.title("My First App")

""" Writing """
# (st.write() will write in markdown)
st.write("Here's our first attempt at using data to create a table:")
# st.dataframe() and st.table() can also be used for displaying data
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

# You can write to your app without calling Streamlit methods
# Streamlit supports "magic commands"
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})
df  # Streamlit automatically writes the variable using st.write()

""" Draw charts and maps """
# Draw a line chart
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# Plot a map
# st.map() display points on a map
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

""" Add interactivity with widgets """
