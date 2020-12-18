import numpy as np
import cv2
from skimage import feature
import dlib
from constants import IMG_DIM


def get_LBP(img):
    """Returns an LBP feature vector of img that is grayscaled'

    Args:
        img : ndarray
            The input image (grayscaled).
    Returns:
        norm_hist : ndarray
            The normalized histogram pattern from LBP of img.

    Source: https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    """

    #Initializing LBP feature for our image
    LBP = feature.local_binary_pattern(img, 24, 8, method="uniform")

    #Create histogram for LBP
    (hist, _) = np.histogram(LBP.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
    hist = hist.astype("float")

    #Normalize histogram
    norm_hist = np.zeros(hist.shape)
    for i in range(len(hist)):
        norm_hist[i] = hist[i] / (hist.sum() + 0.00001)

    return norm_hist

def get_hog(img, tau):
    """Returns a HOG feature vector of img that is grayscaled'

    Args:
        img : ndarray
            The input image (grayscaled).
        tau : int
            The cell height and width.
    Returns:
        hog_feats : ndarray
            The HOG feature vector for img.
    """

    cell_dim = (tau, tau)
    block_size = (2, 2)

    # Dimensions of cropped image to fit cells
    img_crop_dim = (img.shape[1] // cell_dim[1] * cell_dim[1], img.shape[0] // cell_dim[0] * cell_dim[0])

    block_dim = (block_size[1] * cell_dim[1], block_size[0] * cell_dim[0])

    # Initialize hog descriptor
    nbins = 9
    hog = cv2.HOGDescriptor(_winSize=img_crop_dim, _blockSize=block_dim, _blockStride=(cell_dim[1], cell_dim[0]),
                            _cellSize=cell_dim, _nbins=nbins)

    hog_feats = hog.compute(img).flatten()
    return hog_feats

def get_facial_landmarks(img, shape_predictor):
    """Returns a facial landmark feature vector of img that is grayscaled'

    Args:
        img : ndarray
            The input image (grayscaled).
        shape_predictor : dlib.shape_predictor(img, dlib.rectangle)
            The pretrained FL detector that returns a shape that represents landmarks
    Returns:
        FL_feat : ndarray
            The FL feature vector for img.
    """

    # Uses dlibs pretrained facial landmark detector and returns shape that contains
    # 68 x,y coordinates of points of interest
    pred = shape_predictor(img, dlib.rectangle(0, 0, IMG_DIM, IMG_DIM))

    # Convert shape to x and y coordinates
    x_y = np.zeros((68, 2), dtype="int")

    for i in range(0, 68):
        x = pred.part(i).x
        y = pred.part(i).y

        x_y[i] = [x, y]

    FL_feat = x_y.flatten()
    return FL_feat