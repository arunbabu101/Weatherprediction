import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Book1.csv')
    return df

df = load_data()

# Sidebar - Collects user input features into dataframe
st.sidebar.header('User Input Parameters')

def user_input_features():
    temperature = st.sidebar.slider('Temperature (°C)', int(df['Temperature'].min()), int(df['Temperature'].max()), int(df['Temperature'].mean()))
    humidity = st.sidebar.slider('Humidity (%)', int(df['Humidity'].min()), int(df['Humidity'].max()), int(df['Humidity'].mean()))
    wind_speed = st.sidebar.slider('Wind Speed (km/h)', int(df['Wind_Speed'].min()), int(df['Wind_Speed'].max()), int(df['Wind_Speed'].mean()))
    data = {'Temperature': temperature,
            'Humidity': humidity,
            'Wind_Speed': wind_speed}
    features = pd.DataFrame(data, index=[0])
    return features

df_user = user_input_features()

# Main panel
st.subheader('User Input parameters')
st.write(df_user)

# Prepare the dataset
X = df[['Temperature', 'Humidity', 'Wind_Speed']]
y = df['Weather']

# Encode categorical variables
X = pd.get_dummies(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test)

# Example prediction
example_pred = logistic_model.predict(scaler.transform(df_user))
st.subheader('Predicted Weather')
st.write(example_pred[0])
