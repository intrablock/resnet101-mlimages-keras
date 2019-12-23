
import keras
from keras.models import load_model
import keras.backend as K
from keras.utils import get_file
import numpy as np
import cv2
import time

model_file = './resnet101_mlimages_11166_no_top.h5'
model = load_model(model_file)
result = model.predict(np.zeros((1,224,224,3)))
print(result)
# keras
# [[[[0.02104525 0.03589199 0.00509268 ... 3.1072438  0.73230106
#     1.7426963 ]]]]

model_file = './resnet101_mlimages_11166_top.h5'
model = load_model(model_file)

def preprocess(img):
    rawH = float(img.shape[0])
    rawW = float(img.shape[1])
    newH = 256.0
    newW = 256.0
    test_crop = 224.0 

    if rawH <= rawW:
        newW = (rawW/rawH) * newH
    else:
        newH = (rawH/rawW) * newW
    img = cv2.resize(img, (int(newW), int(newH)))
    img = img[int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop),int((newW-test_crop)/2):int((newW-test_crop)/2)+int(test_crop)]
    img = ((img/255.0) - 0.5) * 2.0
    img = img[...,::-1]
    return img

CLASS_INDEX = None
CLASS_INDEX_PATH = 'dictionary_and_semantic_hierarchy.txt'
DICT_URL = 'https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt'
CLASS_INDEX_PATH = get_file(CLASS_INDEX_PATH, DICT_URL)
# idx->(id,name)
def _load_dictionary(dict_file):
    dictionary = dict()
    with open(dict_file, 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            sp = line.rstrip('\n').split('\t')
            idx, cid, name = int(sp[0]), sp[1], sp[3]
            dictionary[idx] = (cid, name)
    return dictionary

def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if CLASS_INDEX is None:
        CLASS_INDEX = _load_dictionary(CLASS_INDEX_PATH)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[i]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

path = 'im_0.jpg'
IMG_URL = 'https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/images/im_0.jpg'
path = get_file(path, IMG_URL)
# inference
raw_img = cv2.imread(path)
start = time.time()
img = preprocess(raw_img)
img = np.expand_dims(img, 0)
result = model.predict(img)
used = time.time() - start
print(decode_predictions(result), used)