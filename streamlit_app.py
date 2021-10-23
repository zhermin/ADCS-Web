import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, time
from PIL import Image

import tensorflow as tf
from keras.models import load_model
from keras.applications import vgg16

#------------------------------------------------------------------------------#

IMG_SIZE = 256
BATCH_SIZE = 16
DEFECT_LIST = ['none', 'chipping']
DEFECT_MAPPING = dict(enumerate(DEFECT_LIST))

#------------------------------------------------------------------------------#

@st.cache
def load_images(image_files):
    for i in range(len(image_files)):
        img = Image.open(image_files[i])
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.asarray(img)[:, :, :3]
        image_files[i] = img

    image_files = np.asarray(image_files)
    image_files = vgg16.preprocess_input(image_files)
    return image_files

@st.cache(allow_output_mutation=True)
def model_load(model_name):
    loaded_model = load_model(os.path.join(model_dir, model_name))
    loaded_model.make_predict_function()
    loaded_model.summary()
    return loaded_model

@st.cache(show_spinner=False)
def predict(imgs):
    with tf.device('/cpu:0'):
        pred = model.predict(imgs, batch_size=1)

    df_pred = pd.DataFrame(pred, columns=DEFECT_LIST)
    df_pred['prediction'] = np.argmax(pred, axis=1)
    df_pred['prediction'] = df_pred['prediction'].map(DEFECT_MAPPING.get)
    return df_pred

#------------------------------------------------------------------------------#

st.write("""
# CNN for Wafer Edge ADC
Latest Results: 99.99% Out of Sample Accuracy (4571/4577)

---
""")

model_dir = os.path.join(os.getcwd(), 'models')
model_paths = glob.glob(os.path.join(model_dir, '*.h5'))
model_names = [full_path.split('\\')[-1] for full_path in model_paths]

model_name = st.sidebar.selectbox(
    'Select Model',
    model_names,
    index=len(model_names)-1,
)

st.write(f"## Model Selected: {model_name}")
st.write("")

model = None
model = model_load(model_name)

if model == None:
    st.write("Model not loaded yet")

#------------------------------------------------------------------------------#

# st.write(f"## Image Upload")

# image_files = st.file_uploader("", type=['png','jpeg','jpg'], accept_multiple_files=True)
# st.write(image_files)

with st.form("image-uploader", clear_on_submit=True):
    image_files = st.file_uploader("IMAGE UPLOADER", type=['png','jpeg','jpg'], accept_multiple_files=True)
    submitted = st.form_submit_button("UPLOAD BATCH")

    if submitted:
        if len(image_files) > 0:
            st.success(f"{len(image_files)} FILE(S) UPLOADED!")
        else:
            st.info("FILES CLEARED")

if len(image_files) > 0:

    # st.write(dir(image_file))
    # file_details = {
    #     "Filename":image_file.name,
    #     "FileType":image_file.type,
    #     "FileSize":image_file.size
    # }
    # st.write(file_details)

    imgs = load_images(image_files)
    # st.image(img, width=IMG_SIZE)

    with st.spinner('Predicting...'): 
        start = time.time()
        df_pred = predict(imgs)

    st.write(f"---")
    st.write(f"## Image Predictions")

    st.write(df_pred)
    # st.write(f'Prediction for this image is: {DEFECT_MAPPING.get(np.argmax(pred, axis=1)[0])} ({np.max(pred):.2%} confident)')
    st.write(f'Runtime: {round((time.time() - start)/60, 2)} mins')