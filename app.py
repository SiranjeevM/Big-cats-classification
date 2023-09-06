
import streamlit as st
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from PIL.image import Image
import numpy as np
st.title("Big cat Image Classification")

st.write("Predict the cat that is being represented in the image.")

model = load_model("model.h5",custom_objects={'KerasLayer':hub.KerasLayer})
labels = {
        0: 'AFRICAN LEOPARD',
        1: 'CARACAL',
        2: "CHEETAH",
        3: "CLOUDED LEOPARD",
        4: "JAGUAR",
        5: "LION",
        6: "OCELOT",
        7:"PUMA",
        8:"SNOW LEOPARD",
        9:"TIGER"
    }



uploaded_file = st.file_uploader(
    "Upload an image of a cat type animal :", type='jpg'
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    image1 = Image.open("9.tiger.jpg")
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]
    image1 = Image.open("9.tiger.jpg")
    st.image(image1, caption="Uploaded Image", use_column_width=True)    
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )
