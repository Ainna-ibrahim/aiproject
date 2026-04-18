import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import random
import string

# Set the title of the page
st.set_page_config(page_title="Student Predictor", layout="centered")

# Add custom CSS for button styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50; /* Green background */
            color: white; /* White text */
            font-size: 18px;
            border-radius: 12px;
            padding: 10px 24px;
            border: none;
            cursor: pointer;
            width: 150px;  /* Adjust width of button */
            margin-top: 10px;
        }

        .stButton>button:hover {
            background-color: #45a049; /* Darker green on hover */
        }

        .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 10px #4CAF50; /* Green border shadow when focused */
        }

        /* Add spacing and style for the header buttons (Sign Up / Login) */
        .stButton>button {
            margin-left: 15px;
            margin-right: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Section for About
st.sidebar.title("About")
st.sidebar.info("AI model predicts student performance based on student inputs.")

# Display the Sign Up and Login buttons at the top
col1, col2 = st.columns([1, 1])  # Create two columns for buttons
with col1:
    if st.button("Sign Up"):
        st.session_state.page = "sign_up"  # Navigate to sign-up page
        st.experimental_rerun()  # Redirect to Sign Up page
with col2:
    if st.button("Login"):
        st.session_state.page = "login"  # Navigate to login page
        st.experimental_rerun()  # Redirect to Login page

# Main UI for the Student Performance Prediction
st.markdown(
    "<h1 style='text-align: center;'>🎓 Student Performance Predictor</h1>",
    unsafe_allow_html=True
)

# If user is logged in or on the main page, show student input form
if "page" not in st.session_state or st.session_state.page == "main":
    # Load the data for prediction
    data = pd.read_csv("data/student-mat.csv")

    # Prepare features and target
    X = data.drop("G3", axis=1)
    y = data["G3"]

    # Convert categorical data
    X = pd.get_dummies(X)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Form for student details input
    st.write("Enter student details:")

    age = st.slider("Age", 15, 22, 18)
    studytime = st.slider("Study Time (1-4)", 1, 4, 2)
    absences = st.slider("Absences", 0, 30, 5)
    g1 = st.slider("First Term Marks (G1)", 0, 20, 10)
    g2 = st.slider("Second Term Marks (G2)", 0, 20, 10)

    # Create input data
    input_data = pd.DataFrame({
        "age": [age],
        "studytime": [studytime],
        "absences": [absences],
        "G1": [g1],
        "G2": [g2]
    })

    # Match columns with training data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Final Marks: {prediction:.2f}")

        if prediction >= 15:
            st.balloons()
            st.info("Excellent Performance 🚀")
        elif prediction >= 10:
            st.info("Average Performance 👍")
        else:
            st.warning("Needs Improvement ⚠️")

# Sign-Up Page
if st.session_state.page == "sign_up":
    st.title("Sign Up")

    # Sign-up form with validation
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    # Validate the form inputs
    if st.button("Next"):
        if not first_name or not last_name or not email or not password or password != confirm_password:
            st.error("Please fill in all fields correctly.")
        else:
            # Save the data to session state for the next page
            st.session_state.first_name = first_name
            st.session_state.last_name = last_name
            st.session_state.email = email
            st.session_state.password = password
            st.session_state.page = "username_suggestions"
            st.experimental_rerun()

# Username Suggestions Page
if st.session_state.page == "username_suggestions":
    st.title("Choose Your Username")

    def generate_username(first_name, last_name):
        username_list = [
            first_name[:3] + last_name + str(random.randint(100, 999)),
            first_name + "_" + last_name,
            first_name[:3] + random.choice(string.ascii_lowercase) + str(random.randint(100, 999)),
        ]
        return username_list

    suggested_usernames = generate_username(st.session_state.first_name, st.session_state.last_name)
    for username in suggested_usernames:
        st.write(username)

    # Option for the user to choose an AI-generated username or create their own
    selected_username = st.text_input("Or create your own username:")
    if selected_username:
        st.session_state.username = selected_username
        st.success(f"Username chosen: {selected_username}")
        
        # Sidebar Section for About
st.sidebar.title("About")
st.sidebar.info("AI model predicts student performance based on student inputs.")

# Sign-Up and Login buttons in the sidebar
sign_up_button = st.sidebar.button("Sign Up")
login_button = st.sidebar.button("Login")

# Add functionality for each button
if sign_up_button:
    st.session_state.page = "sign_up"  # Navigate to sign-up page
    st.experimental_rerun()  # Redirect to Sign Up page

if login_button:
    st.session_state.page = "login"  # Navigate to login page
    st.experimental_rerun()  # Redirect to Login page