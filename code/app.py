import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, time
from PIL import Image

from keras.models import load_model
from keras.applications import vgg16

np.set_printoptions(suppress=True)

IMG_SIZE = 256
DEFECT_LIST = ['none', 'chipping']
DEFECT_MAPPING = dict(enumerate(DEFECT_LIST))

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

@st.cache(allow_output_mutation=True)
def model_load(model_name):
    loaded_model = load_model(os.path.join(model_dir, model_name))
    loaded_model.make_predict_function()
    loaded_model.summary()
    return loaded_model


st.write("""
# Wafer Edge ADC using CNN
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


model = None
model = model_load(model_name)

if model == None:
    st.write("Model not loaded yet")

st.write(f"## Image Upload")

image_file = st.file_uploader("", type=['png','jpeg','jpg'])

if image_file is not None:

    # st.write(dir(image_file))

    # file_details = {
    #     "Filename":image_file.name,
    #     "FileType":image_file.type,
    #     "FileSize":image_file.size
    # }
    # st.write(file_details)

    img = load_image(image_file)
    st.image(img, width=256)

    img = img.resize((256, 256))
    img = np.asarray(img)[:, :, :3]
    img = vgg16.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    st.write(f"---")
    st.write(f"## Single Image Prediction")

    with st.spinner('Predicting...'): 
        start = time.time()
        pred = model.predict(img)
    df_pred = pd.DataFrame(pred, columns=DEFECT_LIST)

    st.write(df_pred)
    st.write(f'Prediction for this image is: {DEFECT_MAPPING.get(np.argmax(pred, axis=1)[0])} ({np.max(pred):.2%} confident)')
    st.write(f'Runtime: {round((time.time() - start)/60, 2)} mins')