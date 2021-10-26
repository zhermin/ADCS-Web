import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, time
from PIL import Image

import tensorflow as tf
from keras.models import load_model
from keras.applications import vgg16

st.set_page_config(layout="wide")

#------------------------------------------------------------------------------#

IMG_SIZE = 256
BATCH_SIZE = 1
DEFECT_LIST = ['none', 'chipping']
DEFECT_MAPPING = dict(enumerate(DEFECT_LIST))

#------------------------------------------------------------------------------#

@st.cache
def load_images(image_files):
    imgs = []
    for i in range(len(image_files)):
        img = Image.open(image_files[i])
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.asarray(img)[:, :, :3]
        imgs.append(img)

    imgs = np.asarray(imgs)
    imgs = vgg16.preprocess_input(imgs)
    return imgs

@st.cache(allow_output_mutation=True)
def model_load(model_name):
    loaded_model = load_model(os.path.join(model_dir, model_name))
    loaded_model.make_predict_function()
    loaded_model.summary()
    return loaded_model

@st.cache(show_spinner=False)
def predict(imgs):
    with tf.device('/cpu:0'):
        pred = model.predict(imgs, batch_size=BATCH_SIZE)
    return pred

#------------------------------------------------------------------------------#

st.write("""
# CNN for Wafer Edge ADC
Latest Results: Model "vgg16_13Oct-1845.h5" achieved 99.99% Out of Sample Accuracy (4571/4577)

---
""")

model_dir = os.path.join(os.getcwd(), 'models')
model_paths = glob.glob(os.path.join(model_dir, '*.h5'))
model_names = [model_path.split('\\')[-1] for model_path in model_paths]

model_name = st.sidebar.selectbox(
    label='Select Model',
    options=model_names,
    index=len(model_names)-1,
)

threshold = st.sidebar.slider(
    label='Select Confidence Percent Threshold',
    min_value=80.00,
    max_value=100.00,
    value=90.00,
    step=0.01,
    format='%.2f'
)
threshold /= 100

max_cols = st.sidebar.slider(
    label='Select Number of Images per Row',
    min_value=1,
    max_value=20,
    value=10,
)

st.write(f"## Model Selected\n `>> {model_name}`")
st.write("")

model = None
model = model_load(model_name)

if model == None:
    st.write("Model not loaded yet")

#------------------------------------------------------------------------------#

with st.form("image-uploader", clear_on_submit=True):
    image_files = st.file_uploader("IMAGE UPLOADER", type=['png','jpeg','jpg'], accept_multiple_files=True)
    submitted = st.form_submit_button("UPLOAD/CLEAR BATCH")

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

    with st.spinner(f'Predicting {len(imgs)} Images...'): 
        start = time.time()
        pred = predict(imgs)

    df_pred = pd.DataFrame(pred, columns=DEFECT_LIST)
    df_pred['confidence'] = np.max(pred, axis=1)
    df_pred['unconfident'] = np.where(df_pred['confidence'] < threshold, True, False)
    df_pred['prediction'] = np.argmax(pred, axis=1)
    df_pred['prediction'] = df_pred['prediction'].map(DEFECT_MAPPING.get)

    #------------------------------------------------------------------------------#

    st.write(f"---")
    st.write(f"## {len(imgs)} Image Predictions")

    st.write(df_pred.style.highlight_max(color='grey', axis=1))
    st.write(f'Runtime: {round((time.time() - start)/60, 2)} mins')

    #------------------------------------------------------------------------------#

    unconfident_rows = df_pred.loc[df_pred['unconfident'] == True]
    unconfident_rows = unconfident_rows.drop('unconfident', axis=1)
    unconfident_idx = unconfident_rows.index

    num_imgs = len(unconfident_idx)
    num_rows = num_imgs//max_cols if num_imgs%max_cols==0 else num_imgs//max_cols+1

    st.write(f"---\n## {num_imgs} Unconfident Predictions (< {threshold:.2%} Confidence)")
    st.write(unconfident_rows)
    st.write("")

    for row in range(num_rows):
        remaining_imgs = num_imgs-max_cols*row
        num_cols = max_cols if remaining_imgs >= max_cols else remaining_imgs
        st_row = st.columns(max_cols)
        for col in range(num_cols):
            image_file = image_files[unconfident_idx[row*max_cols+col]]
            st_row[col].image(image_file)

            selected_row = unconfident_rows.iloc[row*max_cols+col]
            st_row[col].write(f'#### [{unconfident_idx[row*max_cols+col]}] {image_file.name}')
            st_row[col].write(f'###### {selected_row["prediction"]} ({selected_row["confidence"]:.2%})')