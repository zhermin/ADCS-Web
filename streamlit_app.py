#------------------------------ Table of Contents -----------------------------#
""" // 13 November 2021; 600 lines yay; by Zher Min;
|
+-- Library Imports
+-- Page Config
+-- Constants
+-- State Variables
+-- Functions
+-- None Pagination
+-- Sidebar
+-- Introduction
+-- Image Uploader
+-- Load Model
+-- Start Predictions
+-- Data Persistence
\-- Batch Predictions
    |
    +-- Summary
    +-- All Predictions
    +-- None Predictions
    +-- Chipping Predictions
    \-- Unconfident Predictions
"""
#------------------------------- Library Imports ------------------------------#

import streamlit as st
import numpy as np
import pandas as pd
import os, glob, time, re
from PIL import Image

import tensorflow as tf
from keras.models import load_model

#--------------------------------- Page Config --------------------------------#
#-- Sets config for the Streamlit app. Also use some css hacks to get rid of --#
#-- Streamlit's hamburger menu and footer texts.                             --#
#------------------------------------------------------------------------------#

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
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#---------------------------------- Constants ---------------------------------#
#-- Image size of 224 is used since that's the standard for most CNN models. --#
#-- Also, my models were trained using that size, so this is necessary.      --#
#-- My problem statement only has 2 classes: none or chipping.               --#
#------------------------------------------------------------------------------#

IMG_SIZE = 224
BATCH_SIZE = 1
DEFECT_LIST = ['none', 'chipping']
DEFECT_MAPPING = dict(enumerate(DEFECT_LIST))

#------------------------------- State Variables ------------------------------#
#-- Initialise Streamlit state variables for data persistence across reruns. --#
#------------------------------------------------------------------------------#

if 'none_page' not in st.session_state: st.session_state['none_page'] = 0
if 'reset_demo' not in st.session_state: st.session_state['reset_demo'] = True
if 'save_batch' not in st.session_state: st.session_state['save_batch'] = True
if 'df_pred_saved' not in st.session_state: st.session_state['df_pred_saved'] = pd.DataFrame()
if 'df_summary_saved' not in st.session_state: st.session_state['df_summary_saved'] = pd.DataFrame()

#---------------------------------- Functions ---------------------------------#
#-- Functions are defined here. Cached functions from Streamlit help store   --#
#-- return variables into cache. If there are no changes to the output, the  --#
#-- functions are not rerun to speed up the loading of these outputs.        --#
#------------------------------------------------------------------------------#

@st.cache(allow_output_mutation=True, show_spinner=False)
def model_load(model_name):
    """Loads selected machine learning model
    
    Flag 'allow_out_mutation=True' tells Streamlit that the output will change
    and that we are aware of it. This is necessary to load the Keras model. 

    Args:
        model_name (str): User selected model from app's sidebar

    Returns:
        loaded_model (Keras model): Trained machine learning model from my repo
    """

    loaded_model = load_model(os.path.join(model_dir, model_name))
    loaded_model.make_predict_function()
    print(model_name)
    return loaded_model

def load_images(image_files):
    """Loads images uploaded through Streamlit's file_uploader widget

    This is just an intermediate helper function to load and process images.
    Hence, this does not need to be cached.

    Args:
        image_files (list): List of uploaded images

    Returns:
        imgs (np.array): 
            Resized using PILLOW and preprocessed by selected model's 
            preprocessing function and stored in a numpy array
    """

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
    """Performs model prediction on the loaded images

    Once images are loaded and preprocessed by load_images(), feeds them
    into the model for prediction with specified BATCH_SIZE, which is 1
    in most cases, since higher batch sizes require more memory. 

    Args:
        image_files (list): List of uploaded images
    
    Returns:
        pred (np.array): Prediction tensor in numpy array format
    """

    with tf.device('/cpu:0'):
        pred = model.predict(load_images(image_files), batch_size=BATCH_SIZE)
    return pred

def standardise_filename(string):
    """Changes an underscore to a dash at a specific location

    This is another helper function to change one specific character. 
    This is because the filename format for the chipping images are of a 
    specific format. This will not be relevant outside of these images. 
    
    Args:
        string (str): Specific filename of the input images

    Returns:
        string (str): Same string but with an underscore changed to a dash
    """

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
    """Takes in prediction tensor and outputs a bunch of dataframes for analysis

    After getting the prediction tensor from the model, we can analyse the
    results and output a few dataframes for specific functions such as
    viewing chipping or non-chipping images only. 

    Also tries to apply standardise_filename() function to split the filename
    and get the wafer IDs and lot IDs information. 

    Args:
        pred (np.array): Prediction tensor
        threshold (float): 
            User-specified confidence threshold 
            for predictions to cross from sidebar

    Returns:
        df_xxx (pandas dataframe): 
            Bunch of dataframes that have been drilled down
            for different purposes
    """

    df_pred = pd.DataFrame(pred, columns=DEFECT_LIST)
    if demo_mode: df_pred.insert(0, 'filename', [image_file.split('\\')[-1] for image_file in image_files])
    else: df_pred.insert(0, 'filename', [image_file.name for image_file in image_files])

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
    """Converts dataframe to a downloadable csv

    Helper function with caching to generate download buttons for the dataframes

    Args:
        df (pandas dataframe): Any dataframe
    
    Returns:
        a downloadable csv format
    """

    return df.to_csv(index=False, encoding='utf-8')

def display_images(image_files, df, max_cols):
    """Displays images onto the Streamlit app

    This helps arrange all the images onto the app evenly with the
    user-specified max_cols variable to control the size of each image. 

    Args:
        image_files (list): List of uploaded images
        df (pandas dataframe): 
            The corresponding dataframe that the images are
            to be displayed
        max_cols (int): 
            User-specified integer to control number of columns
            to display in each row. In turn affects the size of images. 
    """

    idx = df.index

    num_imgs = len(idx)
    num_rows = num_imgs // max_cols if num_imgs % max_cols == 0 else num_imgs // max_cols + 1

    for row in range(num_rows):
        remaining_imgs = num_imgs-max_cols*row
        num_cols = max_cols if remaining_imgs >= max_cols else remaining_imgs

        st_row = st.columns(max_cols)
        for col in range(num_cols):
            image_file = image_files[idx[row*max_cols+col]]
            image_name = image_file.split('\\')[-1] if demo_mode else image_file.name
            selected_row = df.iloc[row*max_cols+col]

            st_row[col].image(
                Image.open(image_file) if demo_mode else image_file, 
                caption=f'[{idx[row*max_cols+col]}] {selected_row["prediction"]} ({selected_row["confidence"]:.2%}) {image_name if toggle_names else ""}'
            )

#------------------------------- None Pagination ------------------------------#
#-- Some callback functions to enable pagination for none predictions.       --#
#-- This is needed because none predictions are usually in the hundreds.     --#
#------------------------------------------------------------------------------#

def update_none_page(new_none_page):
    st.session_state['none_page'] = new_none_page

def prev_none_page(): 
    updated_page = st.session_state['none_page'] - 1
    if updated_page >= 0: st.session_state['none_page'] = updated_page

def next_none_page(): 
    updated_page = st.session_state['none_page'] + 1
    if updated_page < none_pages: st.session_state['none_page'] = updated_page

#----------------------------------- Sidebar ----------------------------------#
#-- Packs all the user settings to the sidebar.                              --#
#-- The model's preprocessing function is also determined here depending on  --#
#-- which model was selected. My trained models only used either of          --#
#-- 3 backbones, so only 3 of the functions are specified.                   --#
#------------------------------------------------------------------------------#

model_dir = os.path.join(os.getcwd(), 'models')
model_paths = glob.glob(os.path.join(model_dir, '*.h5'))
model_names = sorted([model_path.split('\\')[-1].split('/')[-1] for model_path in model_paths])

model_name = st.sidebar.selectbox(
    label='Select Model',
    options=model_names,
    index=0,
)

PRETRAINED_NAME = model_name.split('_')[0]
if PRETRAINED_NAME == "vgg16": PREPROCESSING_FUNCTION = tf.keras.applications.vgg16.preprocess_input
elif PRETRAINED_NAME == "resnet50v2": PREPROCESSING_FUNCTION = tf.keras.applications.resnet_v2.preprocess_input
elif PRETRAINED_NAME == "mobilenetv2": PREPROCESSING_FUNCTION = tf.keras.applications.mobilenet_v2.preprocess_input

threshold = st.sidebar.slider(
    label='Select Confidence % Threshold',
    min_value=80,
    max_value=100,
    value=95,
    step=1,
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

def update_reset_demo():
    st.session_state['reset_demo'] = True

st.sidebar.write("---")
demo_mode = st.sidebar.checkbox(
    label='Toggle Demo Mode',
    value=True,
    on_change=update_reset_demo,
)

#-------------------------------- Introduction --------------------------------#

st.write("""
# CNN for Wafer Edge ADC
#### // An ML Model Deployment UI MVP
By: `Tam Zher Min`  
Email: `tamzhermin@gmail.com`

*Note: Toggle Demo Mode off to upload your own images*  
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
    * Buttons above every table to download as CSV (Excel-readable file)
    * Quick summary table to highlight the lot IDs and wafer IDs with chipping images
    * Demo Mode to upload 5 chipping and 5 non-chipping images to demo the model for external viewers
    * Data persistence so that users can upload images in batches and still observe previous results
    """)
st.write("---")

st.write(f"## Model Selected\n `>> {model_name}`")
st.write("")

#------------------------------- Image Uploader -------------------------------#
#-- Allows users to upload wafer scans using Streamlit's image uploader.     --#
#-- Demo mode also included for God knows who will ever try this app. lol.   --#
#-- Just uploads 10 images stored in my repo to simulate a user using it.    --#
#------------------------------------------------------------------------------#

def upload_success(image_files):
    st.session_state['none_page'] = 0
    st.success(f"{len(image_files)} IMAGE(S) UPLOADED!")
    st.session_state['save_batch'] = True

with st.form("image-uploader", clear_on_submit=True):
    if demo_mode: 
        submitted_demo = st.form_submit_button("DEMO UPLOAD")
        if submitted_demo: st.session_state['reset_demo'] = False

        if st.session_state['reset_demo']:
            image_files = []
        else:
            image_files = glob.glob(os.path.join(os.getcwd(), 'demo', '*.jpg'))
            upload_success(image_files)
    else: 
        image_files = st.file_uploader(
            "UPLOAD IMAGES TO PREDICT (MAX 500)", 
            type=['png','jpeg','jpg'], 
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("UPLOAD/CLEAR BATCH")
        if submitted:
            if len(image_files) > 0:
                upload_success(image_files)
            else:
                st.info("IMAGES CLEARED")

#---------------------------------- Load Model --------------------------------#

with st.spinner(f'Loading Model...'): 
    model = model_load(model_name)

#------------------------------ Start Predictions -----------------------------#

if (not demo_mode and len(image_files) > 0) or (demo_mode and not st.session_state['reset_demo']):

    with st.spinner(f'Predicting {len(image_files)} Images...'): 
        start = time.time()
        pred = predict(image_files)

    df_pred, df_none, df_chipping, df_unconfident, df_summary = load_df(pred, threshold)

    if not demo_mode and st.session_state['save_batch']:
        st.session_state['df_pred_saved'] = st.session_state['df_pred_saved'].append(df_pred, ignore_index=True)
        st.session_state['df_summary_saved'] = st.session_state['df_summary_saved'].append(df_summary, ignore_index=True)
        st.session_state['save_batch'] = False

#------------------------------ Data Persistence ------------------------------#
#-- Since Streamlit's free servers don't allow too many images uploaded at   --#
#-- once, this allows users to upload images in batches yet have persistence --#
#-- in the dataframes after each batch is uploaded.                          --#
#------------------------------------------------------------------------------#

if len(st.session_state['df_pred_saved']) > 0:
    st.write(f"---")
    st.write(f"## {len(st.session_state['df_pred_saved'])} Persistent Predictions")

    if len(st.session_state['df_summary_saved']) > 0:
        st.write(f'#### Total Summary of Wafer Lots & IDs with Chipping')
        st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(st.session_state['df_summary_saved']),
            file_name=f'Total_Summary.csv',
            mime='text/csv',
        )
        st.dataframe(st.session_state['df_summary_saved'])
    else:
        st.write(f'#### No Chipping Images found so far')

    with st.expander(f'View Total Table of All Predictions'):
        st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(st.session_state['df_pred_saved']),
            file_name=f'Total_Predictions.csv',
            mime='text/csv',
        )
        st.dataframe(st.session_state['df_pred_saved'].style.apply(
            lambda df: ['background: grey' if df['prediction'] == 'chipping' else '' for row in df], 
            axis=1
        ))

#------------------------------ Batch Predictions -----------------------------#
#-- Only displays all these mumbo jumbo if there are actually images         --#
#-- uploaded. Displays the drilled down dataframes for further analysis.     --#
#------------------------------------------------------------------------------#

if (not demo_mode and len(image_files) > 0) or (demo_mode and not st.session_state['reset_demo']):

    #----------------------------------- Summary ----------------------------------#

    st.write(f"---")
    st.write(f"## {len(image_files)} Image Predictions (Runtime: {round((time.time() - start)/60, 2)} mins)")

    if len(df_summary) > 0:
        st.write(f'#### Summary of Wafer Lots & IDs with Chipping')
        st.download_button(
            label='DOWNLOAD',
            data=df_to_csv(df_summary),
            file_name=f'Summary.csv',
            mime='text/csv',
        )
        st.dataframe(df_summary)
    else:
        st.write(f'#### No Chipping Images found in this batch')

    #------------------------------- All Predictions ------------------------------#

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

    #------------------------------ None Predictions ------------------------------#

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

    #---------------------------- Chipping Predictions ----------------------------#

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

    #--------------------------- Unconfident Predictions --------------------------#

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
