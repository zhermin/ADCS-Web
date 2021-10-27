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

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#------------------------------------------------------------------------------#

IMG_SIZE = 256
BATCH_SIZE = 1
DEFECT_LIST = ['none', 'chipping']
DEFECT_MAPPING = dict(enumerate(DEFECT_LIST))

#------------------------------------------------------------------------------#

@st.cache(allow_output_mutation=True)
def model_load(model_name):
    loaded_model = load_model(os.path.join(model_dir, model_name))
    loaded_model.make_predict_function()
    loaded_model.summary()
    return loaded_model

# @st.cache
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

@st.cache(show_spinner=False)
def predict(image_files):
    with tf.device('/cpu:0'):
        pred = model.predict(load_images(image_files), batch_size=BATCH_SIZE)
    return pred

@st.cache
def load_df(pred, threshold):
    df_pred = pd.DataFrame(pred, columns=DEFECT_LIST)
    df_pred['confidence'] = np.max(pred, axis=1)
    df_pred['unconfident'] = np.where(df_pred['confidence'] < threshold, True, False)
    df_pred['prediction'] = np.argmax(pred, axis=1)
    df_pred['prediction'] = df_pred['prediction'].map(DEFECT_MAPPING.get)

    df_chipping = df_pred.loc[df_pred['prediction'] == 'chipping']

    df_unconfident = df_pred.loc[df_pred['unconfident'] == True]
    df_unconfident = df_unconfident.drop('unconfident', axis=1)

    return df_pred, df_chipping, df_unconfident

def highlight_pred(image_files, df, max_cols):
    idx = df.index

    num_imgs = len(idx)
    num_rows = num_imgs//max_cols if num_imgs%max_cols==0 else num_imgs//max_cols+1

    for row in range(num_rows):
        remaining_imgs = num_imgs-max_cols*row
        num_cols = max_cols if remaining_imgs >= max_cols else remaining_imgs

        st_row = st.columns(max_cols)
        for col in range(num_cols):
            image_file = image_files[idx[row*max_cols+col]]
            st_row[col].image(image_file)

            selected_row = df.iloc[row*max_cols+col]
            st_row[col].write(f'#### [{idx[row*max_cols+col]}] {image_file.name}')
            st_row[col].write(f'###### {selected_row["prediction"]} ({selected_row["confidence"]:.2%})')

#------------------------------------------------------------------------------#

st.write("""
# CNN for Wafer Edge ADC
Latest Results: Model "vgg16_13Oct-1845.h5" achieved 99.99% Out of Sample Accuracy (4571/4577)

---
""")

model_dir = os.path.join(os.getcwd(), 'models')
model_paths = glob.glob(os.path.join(model_dir, '*.h5'))
model_names = sorted([model_path.split('\\')[-1].split('/')[-1] for model_path in model_paths])

model_name = st.sidebar.selectbox(
    label='Select Model',
    options=model_names,
    index=len(model_names)-1,
)

threshold = st.sidebar.number_input(
    label='Select Confidence % Threshold',
    min_value=80.00,
    max_value=100.00,
    value=90.00,
    step=0.01,
    # format='%.2f'
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
    image_files = st.file_uploader("UPLOAD IMAGES TO PREDICT", type=['png','jpeg','jpg'], accept_multiple_files=True)
    submitted = st.form_submit_button("UPLOAD/CLEAR BATCH")

    if submitted:
        if len(image_files) > 0:
            st.success(f"{len(image_files)} IMAGE(S) UPLOADED!")
        else:
            st.info("IMAGES CLEARED")

if len(image_files) > 0:

    # st.write(dir(image_file))
    # file_details = {
    #     "Filename":image_file.name,
    #     "FileType":image_file.type,
    #     "FileSize":image_file.size
    # }
    # st.write(file_details)

    # imgs = load_images(image_files)

    with st.spinner(f'Predicting {len(image_files)} Images...'): 
        start = time.time()
        pred = predict(image_files)

    df_pred, df_chipping, df_unconfident = load_df(pred, threshold)

    #------------------------------------------------------------------------------#

    st.write(f"---")
    st.write(f"## {len(image_files)} Image Predictions")

    st.dataframe(df_pred.style.highlight_max(color='grey', axis=1))
    st.write(f'Runtime: {round((time.time() - start)/60, 2)} mins')

    #------------------------------------------------------------------------------#

    st.write(f"---\n## {len(df_chipping)} Chipping Predictions")
    st.dataframe(df_chipping)
    st.write("")
    highlight_pred(image_files, df_chipping, max_cols)

    #------------------------------------------------------------------------------#

    st.write(f"---\n## {len(df_unconfident)} Unconfident Predictions (< {threshold:.2%} Confidence)")
    st.dataframe(df_unconfident)
    st.write("")
    highlight_pred(image_files, df_unconfident, max_cols)