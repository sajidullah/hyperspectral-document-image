# hyperspectral-document-image
INK MISMATCH DETECTION IN HYPERSPECTRAL DOCUMENT IMAGES 
Ink Mismatch Detection using Convolutional Neural Networks is proposed in this work.
CNN is trained with the Spectral Response Vectors (SRVs) of all ink pixels in hyperspectral images from the UWA WIHSI database. The SRVs are resized to form matrices to make them compatible with the CNN.
Pooling is not required in the proposed system because the size of input images and feature maps is v. small.
CNN Architecture-V with four convolutional layers and around 50k learn parameters gives the best accuracy.
The trained CNN gives excellent classification results (98.7% accuracy) on Blue Ink SRVs and also performs very well (89.4% accuracy) on Black Ink SRVs.
Promising segmentation results are achieved using the proposed system on all types of mixed ink combinations.
CNN trained with Blue Ink SRVs classifies both Blue and Black Ink SRVs which implies that CNN extracts generic features from the dataset.
The proposed system can be modified to use unsupervised deep learning to overcome the limitation of prior knowledge. 
