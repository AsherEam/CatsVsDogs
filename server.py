import streamlit as st
from fastai.vision.all import *
import urllib.request

st.title("Cat vs. Dog Classifier")
st.text("Built by Asher Eamranond")

def label_func(f): return f[0].isupper()
# Load our pre-trained model
model = load_learn("my_model.pkl")

#Define a function to make predictions
def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = model.predict(img)
    likelihood_is_cat = outputs[1].item()
    print(likelihood_is_cat)

    if likelihood_is_cat > 0.98:
        return "cat"
    elif likelihood_is_cat < 0.2:
        return "dog"
    else:
        return "not sure, try another picture"

uploaded_file = st.file_uploader("Choose and image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None: # if we upload the file
    # display the uploaded image
    st.image(uploaded_file, caption="Upload Image", use_column_width=True)

    if st.button("Predict:"): # If button is pressed
        prediction = predict(uploaded_file)
        st.write(prediction)

# Open terminal on bottom
# Type streamlit run server.py
