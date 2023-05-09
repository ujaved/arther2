from collections import defaultdict
import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image, ImageOps, ImageFilter
from streamlit_image_coordinates import streamlit_image_coordinates
from sklearn.cluster import KMeans
from collections import defaultdict
import requests
import base64
import io

stability_engine_id = "stable-diffusion-xl-beta-v2-2-2"
stability_api_host = os.getenv('API_HOST', 'https://api.stability.ai')
stability_api_key = os.getenv("STABILITY_API_KEY")


@st.cache_data(show_spinner="Generating drawing...")
def getImageFromText(prompt: str):
    response = requests.post(
        f"{stability_api_host}/v1/generation/{stability_engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {stability_api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": f"a brightly colored drawing showing {prompt}",
                    "weight": 1
                },
                {
                    "text": f"person making a drawing",
                    "weight": -50
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
        },
    )
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    im = data["artifacts"][0]
    return base64.b64decode(im["base64"])


@st.cache_data(show_spinner='')
def auto_canny(image, sigma=0.33):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)

    edges = 255 - edges 
    # to increase contrast
    edges = cv2.createCLAHE().apply(edges)
    return edges

# image is an np array
@st.cache_data(show_spinner='')
def getKMeans(image, number_of_colors):
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)
    clf = KMeans(n_clusters=number_of_colors, n_init='auto')
    labels = clf.fit_predict(modified_image)
    # clusters is a mapping from cluster label to the list of pixel indices having that color
    clusters = dict()
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[int(label)] = []
        clusters[int(label)].append(i)
    return labels, clusters, np.array(clf.cluster_centers_, dtype='uint8')



def promptCallback():
    if st.session_state.prompt:
        image = Image.open(io.BytesIO(
            getImageFromText(st.session_state.prompt)))
        clustered = np.array(image)
        st.session_state.orig_shape = clustered.shape
        labels, clusters, cluster_centers = getKMeans(clustered, 10)
        st.session_state.color_labels = labels
        st.session_state.color_clusters = clusters
        st.session_state.cluster_centers = cluster_centers
        gs_image = image.convert('L')
        image_to_color = auto_canny(np.array(gs_image))
        image_to_color = cv2.cvtColor(image_to_color, cv2.COLOR_GRAY2RGB)
        image_to_color_flattened = image_to_color.reshape((-1, 3))
        st.session_state.image_to_color_flattened = image_to_color_flattened
        st.session_state.colored_regions = set()
        st.session_state.coords = None

        if 'num_images' in st.session_state:
            st.session_state.num_images += 1
        else:
            st.session_state.num_images = 1


st.set_page_config(
    page_title="ColArther",
    page_icon="👋",
)

st.sidebar.markdown('<p class="font">ColArther</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
    st.write("""
      Coloring white regions        
     """)

# each image_history tuple is (image, prompt)
if 'image_history' in st.session_state:
  for p in st.session_state.image_history:
    st.image(p[0], caption=p[1])
else:
    st.session_state.image_history = []

pval = ''
if 'prompt' in st.session_state:
    pval = st.session_state.prompt
st.text_input('What would you like to see in your drawing?',
              value=pval, key='prompt', on_change=promptCallback)

if 'image_to_color_flattened' in st.session_state:
    image_to_color_flattened = st.session_state.image_to_color_flattened
    image = Image.fromarray(
        image_to_color_flattened.reshape(st.session_state.orig_shape))
    streamlit_image_coordinates(image, key="coords")
    col_progress = int(
        100*(len(st.session_state.colored_regions)/len(st.session_state.color_clusters)))
    st.progress(col_progress, text=f'{col_progress}% completed')
    if col_progress >= 100:
        st.balloons()
        if len(st.session_state.image_history) < st.session_state.num_images:  
          st.session_state.image_history.append((image, st.session_state.prompt))
    elif st.session_state.coords:
        idx = st.session_state.coords["y"]*st.session_state.orig_shape[0] + st.session_state.coords["x"]
        if st.session_state.color_labels[idx] not in st.session_state.colored_regions:
            pixels = st.session_state.color_clusters[st.session_state.color_labels[idx]]
            for pIdx in pixels:
                image_to_color_flattened[pIdx] = st.session_state.cluster_centers[st.session_state.color_labels[idx]]
            st.session_state.colored_regions.add(
                st.session_state.color_labels[idx])
            st.experimental_rerun()
