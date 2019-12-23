# Keras | Resnet-101 model pre-trained on Tencent ML-Images

CNN model pre-trained on [Tencent ML-Images](https://github.com/Tencent/tencent-ml-images) can be directly used for image classification in open domain, and is also a good starting point for transfer learning.

The Keras model is obtained by converting the tensorflow [checkpoint](https://github.com/Tencent/tencent-ml-images#checkpoints) provided by the authors, with  Microsoft's model conversion framework [MMdnn](https://github.com/microsoft/MMdnn), and a little bit of manual work.

More details about the model can be found in the original repo.

# Models
* resnet101_mlimages_11166_top.h5 [baidu link](https://pan.baidu.com/s/1OeEgu09iCcrJl5rx86D0_Q)
* resnet101_mlimages_11166_no_top.h5 [baidu link](https://pan.baidu.com/s/1oiqQmVyoGfg8V5WiRqA-gQ)

# Conversion
The model is converted on Win10 with
* Tensorflow 1.9
* Keras 2.2.4
* mmdnn 0.2.5

1. Run mmdnn command to convert the ckpt, but errors occur when converting bn's mean and var.
```
mmconvert -sf tensorflow -in ckpt-resnet101-mlimages/model.ckpt.meta -iw ckpt-resnet101-mlimages/model.ckpt --dstNodeName global_pool/Mean -df keras -om tf_resnet
```
2. Replace tensorflow_parser.py in mmdnn/conversion/tensorflow with [tensorflow_parser_rev.py](tensorflow_parser_rev.py). Please note that this is not a general fix.
3. Run the command again, and it should be converted to IR (Intermediate Representation) successfully, but getting error when saving to Keras.
4. Fix the network in the auto-generated python file. See the difference between [tf_resnet.py](tf_resnet.py) and [tf_resnet_autogen.py](tf_resnet_autogen.py).
5. Use the generated Python class to convert IR to Keras. 
```
python3 tf_resnet.py -i generated.npy -o resnet101_mlimages_11166_no_top.h5
```

# Test
Converted model is tested to verify the correctness. The result of `np.zeros((1,224,224,3))` with no_top model on my platform is as follows:
```
# tf
# [[[[0.02104514, 0.03589169, 0.00509045, ..., 3.1072445 ,
#           0.7322991 , 1.742697  ]]]]
# keras
# [[[[0.02104525 0.03589199 0.00509268 ... 3.1072438  0.73230106
#     1.7426963 ]]]]
```
Include_top model can be used for classification.
The result of the following image is:
<img src="https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/images/im_0.jpg" width="25%" height="25%">
```
[[('n12425281', 'liliaceous plant', 0.13314185), ('n13134302', 'bulbous plant', 0.10677312), ('n12459629', 'star-of-Bethlehem', 0.10307251), ('n12480456', 'tuberose, Polianthes tuberosa', 0.060289863), ('n13121544', 'aquatic plant, water plant, hydrophyte, hydrophytic plant', 0.058853388)]]
```
Please refer to [test_keras.py](test_keras.py).