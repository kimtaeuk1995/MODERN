import io
import os
import pathlib
import pickle
from tempfile import NamedTemporaryFile

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn import preprocessing
from tensorflow.keras.preprocessing import image

from utils import *

st.set_page_config(layout="wide")

# V_1_norm = pickle.load(open("V_1_norm.p", "rb"))
# all_paths = pickle.load(open("all_paths.p", "rb"))
# file_mapping = pickle.load(open("file_mapping.p", "rb"))
data_dir = pathlib.Path("images")
result = pickle.load(open("result.p", "rb"))
IMG_SIZE = (224, 224)
batch_size = 50

preprocess_input = tf.keras.applications.vgg16.preprocess_input


weights_file = "vgg16_furniture_classifier_1129.h5"

base_model = tf.keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

x = base_model.output
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
x = tf.keras.layers.Dropout(0.2)(x)
# x = tf.keras.layers.Dense(1024, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.2*0.5)(x)
x = tf.keras.layers.Dense(11, activation="softmax")(x)
model = tf.keras.Model(inputs=base_model.input, outputs=x)

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["accuracy"],
)

model.load_weights(weights_file)

st.title("Furniture Collection Recommender")
st.markdown(
    """
    This is an image-based product collection recommender that pairs user-inputted product with 
    other visually compatible product collection. Use the slider to see which furnitures you would like to see recommedations for
    """
)

st.sidebar.subheader("Pick type of furniture you want to be recommended")
ref_option = st.sidebar.selectbox(
    "Choose furniture type:",
    (
        "chairs",
        "coffee tables",
        "console tables",
        "dining chair",
        "dining tables",
        "end tables",
        "lamp",
        "ottomans",
        "rug",
        "sofa",
        "tv stand",
    ),
    key="ref",
)

Data_root = "file_mapping"
ref_path_dir = os.path.join(Data_root, ref_option)
all_reg_path = [
    os.path.join(ref_path_dir, fname) for fname in sorted(os.listdir(ref_path_dir))
]


ref_id = st.sidebar.text_input("Enter preferred number of recommendations", "5")
assert ref_id.isnumeric(), "Please enter a number"


def similarity_search(V, v_query, file_mapping, n_results=int(ref_id) + 1):
    v_query = np.expand_dims(v_query, axis=0)
    d = V.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(V))
    distances, closest_indices = index.search(v_query, n_results)
    distances = distances.flatten()
    closest_indices = closest_indices.flatten()
    closest_paths = [file_mapping[idx] for idx in closest_indices]
    # query_img = get_concatenated_images([file_mapping[query_idx]])
    results_img = get_concatenated_images(closest_paths)
    return closest_paths, results_img


# def image_upload(img, target_size):
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return img, x
def image_upload(img, target_size):
    img = ImageOps.fit(img, target_size, Image.ANTIALIAS)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


fc1_extractor = tf.keras.Model(
    inputs=model.input, outputs=model.get_layer("fc1").output
)
fc2_extractor = tf.keras.Model(
    inputs=model.input, outputs=model.get_layer("fc2").output
)

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])


if file is None:
    st.text("Please upload an image file")
else:
    img = Image.open(file)
    img_show = img.resize((288, 288))
    compare = preprocessing.normalize(
        fc1_extractor.predict(image_upload(img, model.input_shape[1:3])[1]), norm="l2",
    ).reshape(4096,)

    closest_paths, results = similarity_search(
        pickle.load(open(all_reg_path[0], "rb")),
        compare,
        pickle.load(open(all_reg_path[1], "rb")),
    )

    st.subheader("Uploaded Furniture")
    st.image(img_show, width=None)

    # st.subheader("Recommendations")
    # st.image(results)
    # grid = st.grid()

    for k, i in enumerate(closest_paths, 1):
        if k + 1 <= len(closest_paths):

            col1, col2, col3 = st.beta_columns(3)
            col1.subheader("Recommendation {}".format(k))
            col1.image(get_concatenated_images(closest_paths[k : k + 1]))
            col2.subheader("Website Link")

            link = result[
                result["id"] == (closest_paths[k].split("/")[2].split(".")[0])
            ]["website_link"].values[0]

            col2.markdown(
                link, unsafe_allow_html=True,
            )
            col3.subheader("Price")
            price = result[
                result["id"] == (closest_paths[k].split("/")[2].split(".")[0])
            ]["prices"].values[0]
            col3.markdown(price)

            # row.write(st.image(get_concatenated_images(closest_paths[k : k + 1])))
            # subgird = row.grid()
            # row2 = subgrid.row()
            # row2.write(
            #     st.markdown(
            #         result[result["id"] == (i.split("/")[2].split(".")[0])][
            #             ["website_link", "prices"]
            #         ]["website_link"],
            #         unsafe_allow_html=True,
            #     )
            # )
            # row2.write(
            #     result[result["id"] == (i.split("/")[2].split(".")[0])][
            #         ["website_link", "prices"]
            #     ]["prices"]
            # )
            # st.image(get_concatenated_images(closest_paths[k : k + 1]))
            # st.markdown(
            #     result[result["id"] == (i.split("/")[2].split(".")[0])][
            #         ["website_link", "prices"]
            #     ]["website_link"],
            #     unsafe_allow_html=True,
            # )
        # st.text(
        #     result[result["id"] == (i.split("/")[2].split(".")[0])][
        #         ["website_link", "prices"]
        #     ]
        # )
