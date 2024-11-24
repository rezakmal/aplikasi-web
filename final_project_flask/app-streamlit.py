import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from source.info_hewan import get_kucing_info, get_anjing_info

# Load model and class names
model = load_model("models/model.h5")
loaded_data = np.load("models/model_and_classes_english.npy", allow_pickle=True).item()
class_names = loaded_data["class_names"]

# Streamlit app
st.title("Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img = img.resize((224, 224))  # Resize to (150, 150)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Predict
    if st.button("Classify"):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get index of the highest probability
        predicted_class = class_names[predicted_class_index]  # Get the class name
        st.write(f"This is **{predicted_class}**!")
        if (predicted_class == "cat"):
            
            # Dapatkan informasi kucing
            info_kucing = get_kucing_info()
            
            st.header("Apa itu Kucing?")
            for about in info_kucing["about"]:
                st.write(about)

            # Tampilkan informasi di Streamlit
            st.header("**Jenis-jenis Kucing:**")
            for jenis in info_kucing["jenis"]:
                st.write(jenis)

            st.header("**Makanan Kucing:**")
            for makanan in info_kucing["makanan"]:
                st.write(makanan)

        elif(predicted_class == "dog"):
                        
            # Dapatkan informasi anjing
            info_anjing = get_anjing_info()
            
            st.header("Apa itu Anjing?")
            for about in info_anjing["about"]:
                st.write(about)

            # Tampilkan informasi di Streamlit
            st.header("**Jenis-jenis Anjing:**")
            for jenis in info_anjing["jenis"]:
                st.write(jenis)

            st.header("**Makanan Anjing:**")
            for makanan in info_anjing["makanan"]:
                st.write(makanan)
                
        elif(predicted_class == "horse"):            
            st.header("SKIBIDIIII")