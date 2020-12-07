import os
import zipfile

import wget


def load_data():
    if not os.path.exists("data"):
        os.mkdir("data")
        wget.download(
            "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
            out="data/",
        )
        wget.download(
            "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip",
            out="data/",
        )
        wget.download(
            "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip",
            out="data/",
        )

        local_zip = "data/horse-or-human.zip"
        zip_ref = zipfile.ZipFile(local_zip, "r")
        zip_ref.extractall("data/training")
        zip_ref.close()

        local_zip = "data/validation-horse-or-human.zip"
        zip_ref = zipfile.ZipFile(local_zip, "r")
        zip_ref.extractall("data/validation")
        zip_ref.close()
