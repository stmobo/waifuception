# Waifuception

This is a work-in-progress project aimed towards training a neural network to perform image classification with anime-styled art and images.
My current plan is to perform transfer learning using an Inception-V3 model pretrained on Imagenet.
My initial results are promising, though currently the model appears to have issues converging beyond a few epochs of training.

Currently, the raw data for training is being pulled from the Danbooru2018 dataset, with some filtering and preprocessing applied.

Scripts included:
 - `filter_tags.py` : Processes the metadata included with the Danbooru2018 dataset to select images and also prepare tag vectors.
 - `merge_metadata.py` : Combines the separate output files from `filter_tags` into a single compressed CSV file.
 - `generate_file_list.py` : Outputs a flat textfile containing all filenames selected to be used for training.
    Can be used with `rsync` to only download dataset images required for training.
 - `filter_to_received.py` : Used with partially-downloaded datasets. Filters out undownloaded images from the training dataset.
 - `waifuception/preprocess_images.py` : Compiles a downloaded set of images and CSV metadata to a `TFRecords` file, suitable for training.
 - `waifuception/train.py` : Actually carries out training the neural net.
 - `waifuception/test-predictions.py` : Computes and displays label predictions as generated using saved network weights.
