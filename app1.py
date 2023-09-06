import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import PIL.Image as Image
import tensorflow_hub as hub

# Load the pre-trained ResNet model from TensorFlow Hub
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(resnet_url, trainable=False)

# Define a function to create your custom model
def create_custom_model():
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
    ])
    return model

# Create an instance of the custom model
model = create_custom_model()

# Load the pre-trained weights of your custom model
model.load_weights("model.h5")

# Define a function to make predictions
def predict_image(image_path):
    pred_image = Image.open(image_path).resize((224, 224))
    pred_image = np.array(pred_image) / 255.0
    pred_image = np.expand_dims(pred_image, axis=0)
    result = model.predict(pred_image)
    max_index = np.argmax(result)
    return max_index

# Streamlit app
st.title("Wildlife Classifier")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            max_index = predict_image(uploaded_image)
            classes = [
                "AFRICAN LEOPARD",
                "CARACAL",
                "CHEETAH",
                "CLOUDED LEOPARD",
                "JAGUAR",
                "LION",
                "OCELOT",
                "PUMA",
                "SNOW LEOPARD",
                "TIGER",
            ]
            st.write("Prediction:", classes[max_index])
        except Exception as e:
            st.error("An error occurred during prediction.")
