import streamlit as st
import numpy as np
import pandas as pd
import os, glob, time, re
from PIL import Image

import tensorflow as tf
from keras.models import load_model

#---------------------------------- Page Config -------------------------------#

st.set_page_config(
    page_title="Wafer Edge ADC",
    page_icon="wafer.png",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None,
    }
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#----------------------------------- Constants --------------------------------#

IMG_SIZE = 224
BATCH_SIZE = 1
DEFECT_LIST = ['none', 'chipping']
DEFECT_MAPPING = dict(enumerate(DEFECT_LIST))

#----------------------------------- Functions --------------------------------#

@st.cache(allow_output_mutation=True)
def model_load(model_name):
    loaded_model = load_model(os.path.join(model_dir, model_name))
    loaded_model.make_predict_function()
    loaded_model.summary()
    return loaded_model

def load_images(image_files):
    imgs = []
    for i in range(len(image_files)):
        img = Image.open(image_files[i])
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.asarray(img)[:, :, :3]
        imgs.append(img)

    imgs = np.asarray(imgs)
    imgs = PREPROCESSING_FUNCTION(imgs)
    return imgs

@st.cache(show_spinner=False)
def predict(image_files):
    with tf.device('/cpu:0'):
        pred = model.predict(load_images(image_files), batch_size=BATCH_SIZE)
    return pred

def standardise_filename(string):
    if '-' not in string:
        try:
            where = [m.start() for m in re.finditer('_', string)][-5]
            before = string[:where]
            after = string[where:]
            after = after.replace('_', '-', 1)
            string = before + after
        except:
            pass
    return string

@st.cache
def load_df(pred, threshold):
    df_pred = pd.DataFrame(pred, columns=DEFECT_LIST)
    df_pred.insert(0, 'filename', [image_file.name for image_file in image_files])

    try:
        df_pred['filename'] = df_pred['filename'].apply(standardise_filename)
        df_pred[['lot ID', 'wafer ID']] = df_pred['filename'].str.split('-', expand=True)
        df_pred['lot ID'] = df_pred['lot ID'].str.split('_').str[-1]
        df_pred['wafer ID'] = '#' + df_pred['wafer ID'].str.split('_').str[0]
        df_pred = df_pred[['filename', 'lot ID', 'wafer ID', 'none', 'chipping']]
    except: # skip creating lot ID and wafer ID columns if the filenames are invalid
        pass

    df_pred['confidence'] = np.max(pred, axis=1)
    df_pred['unconfident'] = np.where(df_pred['confidence'] < threshold, True, False)
    df_pred['prediction'] = np.argmax(pred, axis=1)
    df_pred['prediction'] = df_pred['prediction'].map(DEFECT_MAPPING.get)

    df_none = df_pred.loc[df_pred['prediction'] == 'none']
    df_chipping = df_pred.loc[df_pred['prediction'] == 'chipping']

    df_unconfident = df_pred.loc[df_pred['unconfident'] == True]
    df_unconfident = df_unconfident.drop('unconfident', axis=1)

    df_summary = df_chipping.groupby(['lot ID', 'wafer ID'])['filename'].nunique()
    df_summary = df_summary.reset_index()
    df_summary = df_summary.rename(columns={'filename':'no. chipping images'})

    return df_pred, df_none, df_chipping, df_unconfident, df_summary

@st.cache
def df_to_csv(df):
    return df.to_csv(index=False, encoding='utf-8')

def display_images(image_files, df, max_cols):
    idx = df.index

    num_imgs = len(idx)
    num_rows = num_imgs // max_cols if num_imgs % max_cols == 0 else num_imgs // max_cols + 1

    for row in range(num_rows):
        remaining_imgs = num_imgs-max_cols*row
        num_cols = max_cols if remaining_imgs >= max_cols else remaining_imgs

        st_row = st.columns(max_cols)
        for col in range(num_cols):
            image_file = image_files[idx[row*max_cols+col]]
            selected_row = df.iloc[row*max_cols+col]

            st_row[col].image(
                image_file, 
                caption=f'[{idx[row*max_cols+col]}] {selected_row["prediction"]} ({selected_row["confidence"]:.2%}) {image_file.name if toggle_names else ""}'
            )

#------------------------------- None Pagination ------------------------------#

if 'none_page' not in st.session_state:
    st.session_state['none_page'] = 0

def update_none_page(new_none_page):
    st.session_state['none_page'] = new_none_page

def prev_none_page(): 
    updated_page = st.session_state['none_page'] - 1
    if updated_page >= 0: st.session_state['none_page'] = updated_page

def next_none_page(): 
    updated_page = st.session_state['none_page'] + 1
    if updated_page < none_pages: st.session_state['none_page'] = updated_page

#------------------------------------ Sidebar ---------------------------------#

model_dir = os.path.join(os.getcwd(), 'models')
model_paths = glob.glob(os.path.join(model_dir, '*.h5'))
model_names = sorted([model_path.split('\\')[-1].split('/')[-1] for model_path in model_paths])

model_name = st.sidebar.selectbox(
    label='Select Model',
    options=model_names,
    index=0,
)

PRETRAINED_NAME = model_name.split('_')[0]
if PRETRAINED_NAME == "vgg16":
    PREPROCESSING_FUNCTION = tf.keras.applications.vgg16.preprocess_input
elif PRETRAINED_NAME == "resnet50v2":
    PREPROCESSING_FUNCTION = tf.keras.applications.resnet_v2.preprocess_input
elif PRETRAINED_NAME == "mobilenetv2":
    PREPROCESSING_FUNCTION = tf.keras.applications.mobilenet_v2.preprocess_input

threshold = st.sidebar.slider(
    label='Select Confidence % Threshold',
    min_value=80,
    max_value=100,
    value=95,
    step=1,
    # format='%.2f'
) / 100

max_cols = st.sidebar.slider(
    label='Select Number of Images per Row',
    min_value=1,
    max_value=20,
    value=10,
)

max_per_page = st.sidebar.slider(
    label='Select Number of Images per Page',
    min_value=10,
    max_value=100,
    value=40,
    step=10,
)

toggle_names = st.sidebar.checkbox(
    label='Show Image Names',
    value=False,
)

#--------------------------------- Introduction -------------------------------#

st.write("""
# CNN for Wafer Edge ADC
#### // An ML Model Deployment UI MVP
By: `Tam Zher Min`  
Email: `tamzhermin@gmail.com`

*Disclaimer: App is not optimized for performance nor integrated with any internal company infrastructure*  
"""
)

with st.expander(f'Read App Details'):
    st.write("""
    ## Latest Model Updates
    * Previous Results: Model `vgg16_13Oct-1845.h5` achieved 99.99% Out of Sample Accuracy (4571/4577)
    * Update: Model `mobilenetv2_3Nov-1408.h5` achieved 99.57% OOS Accuracy but runs >3x faster than VGG16 models

    ## Features
    *Note: If the website crashes (due to out of memory issues), do contact me to reboot the app*

    * Upload up to 500 (recommended) images at a time for prediction
    * Settings at the Left Sidebar
        * Select a trained model - they vary in accuracy and speed depending on the backbone (eg. VGG16, MobileNetV2, etc.)
        * Select the percent threshold that predictions must meet to be considered a 'confident' prediction
        * Select the number of images per row and per page and toggle image names for best viewing experience
    * Sort by a particular column by clicking on the column name in a table
    * For None predictions, jump to pages if there are a lot of predictions (press Enter after typing a page number and click 'GO') or use the Prev/Next buttons to navigate
    * [NEW] Added buttons above every table to download as CSV (Excel-readable file)
    * [NEW] Added a quick summary table to highlight the lot IDs and wafer IDs with chipping images
    """)
st.write("---")

#---------------------------------- Load Model ------------------------------==#

st.write(f"## Model Selected\n `>> {model_name}`")
st.write("")

model = model_load(model_name)

#-------------------------------- Image Uploader ------------------------------#

with st.form("image-uploader", clear_on_submit=True):
    image_files = st.file_uploader(
        "UPLOAD IMAGES TO PREDICT (MAX 500)", 
        type=['png','jpeg','jpg'], 
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("UPLOAD/CLEAR BATCH")

    if submitted:
        if len(image_files) > 0:
            st.session_state['none_page'] = 0
            st.success(f"{len(image_files)} IMAGE(S) UPLOADED!")
        else:
            st.info("IMAGES CLEARED")

#---------------------------------- PREDICTIONS  ------------------------------#

if len(image_files) > 0:

    with st.spinner(f'Predicting {len(image_files)} Images...'): 
        start = time.time()
        pred = predict(image_files)

    df_pred, df_none, df_chipping, df_unconfident, df_summary = load_df(pred, threshold)

    #----------------------------------- Summary ---------------------------------#

    st.write(f"---")
    st.write(f"## {len(image_files)} Image Predictions (Runtime: {round((time.time() - start)/60, 2)} mins)")

    st.write(f'#### Summary of Wafer Lots & IDs with Chipping')
    if len(df_summary) > 0: 
        st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(df_summary),
            file_name=f'Summary.csv',
            mime='text/csv',
        )
    st.dataframe(df_summary)

    #-------------------------------- All Predictions -----------------------------#

    with st.expander(f'View Table of All Predictions'):
        st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(df_pred),
            file_name=f'All_Predictions.csv',
            mime='text/csv',
        )
        st.dataframe(df_pred.style.apply(
            lambda df: ['background: grey' if df['prediction'] == 'chipping' else '' for row in df], 
            axis=1
        ))

    #------------------------------- None Predictions -----------------------------#

    with st.expander(f'View {len(df_none)} None Predictions'):
        if len(df_none) > 0: st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(df_none),
            file_name=f'None_Predictions.csv',
            mime='text/csv',
        )
        st.dataframe(df_none.style.apply(
            lambda df: ['background: grey' if df['unconfident'] == True else '' for row in df],
            axis=1
        ))
        st.write("")

        none_pages = len(df_none) // max_per_page \
                    if len(df_none) % max_per_page == 0 \
                    else len(df_none) // max_per_page + 1
        if none_pages > 1:
            page_row = st.columns([5,1,1,1])
            with page_row[0]:
                new_none_page = int(st.number_input(
                    label=f"Page {st.session_state['none_page']+1}/{none_pages} (max {max_per_page} per page)",
                    min_value=1,
                    max_value=none_pages,
                    value=int(st.session_state['none_page'])+1,
                    step=1,
                ))-1
            with page_row[1]:
                st.write("&nbsp;")
                st.button(
                    label="GO",
                    on_click=update_none_page,
                    args=(new_none_page,),
                )
            with page_row[2]:
                st.write("&nbsp;")
                st.button("Prev", on_click=prev_none_page)
            with page_row[3]:
                st.write("&nbsp;")
                st.button("Next", on_click=next_none_page)

            display_images(
                image_files, 
                df_none.iloc[
                    st.session_state['none_page']*max_per_page 
                    : (st.session_state['none_page']+1)*max_per_page
                ], 
                max_cols
            )

        else:
            display_images(image_files, df_none, max_cols)

    #----------------------------- Chipping Predictions ---------------------------#

    with st.expander(f'View {len(df_chipping)} Chipping Predictions'):
        if len(df_chipping) > 0: st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(df_chipping),
            file_name=f'Chipping_Predictions.csv',
            mime='text/csv',
        )
        st.dataframe(df_chipping.style.apply(
            lambda df: ['background: grey' if df['unconfident'] == True else '' for row in df],
            axis=1
        ))
        st.write("")
        display_images(image_files, df_chipping, max_cols)

    #-------------------------- Unconfident Predictions ---------------------------#

    with st.expander(f'View {len(df_unconfident)} Unconfident Predictions (< {threshold:.2%} Confidence)'):
        if len(df_unconfident) > 0: st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(df_unconfident),
            file_name=f'Unconfident_Predictions.csv',
            mime='text/csv',
        )
        st.dataframe(df_unconfident)
        st.write("")
        display_images(image_files, df_unconfident, max_cols)