# DIgit-Recognizer

# This is the code for a Kaggle compitition called DIgit-Recognizer
I use a easy CNN to train this model

# This is the structure of the model
________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_28 (Conv2D)           (None, 26, 26, 32)        320       
_________________________________________________________________
activation_46 (Activation)   (None, 26, 26, 32)        0         
_________________________________________________________________
batch_normalization_28 (Batc (None, 26, 26, 32)        104       
_________________________________________________________________
max_pooling2d_28 (MaxPooling (None, 13, 13, 32)        0         
_________________________________________________________________
zero_padding2d_19 (ZeroPaddi (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 13, 13, 48)        13872     
_________________________________________________________________
activation_47 (Activation)   (None, 13, 13, 48)        0         
_________________________________________________________________
batch_normalization_29 (Batc (None, 13, 13, 48)        52        
_________________________________________________________________
max_pooling2d_29 (MaxPooling (None, 6, 6, 48)          0         
_________________________________________________________________
zero_padding2d_20 (ZeroPaddi (None, 8, 8, 48)          0         
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 7, 7, 64)          12352     
_________________________________________________________________
activation_48 (Activation)   (None, 7, 7, 64)          0         
_________________________________________________________________
batch_normalization_30 (Batc (None, 7, 7, 64)          28        
_________________________________________________________________
max_pooling2d_30 (MaxPooling (None, 3, 3, 64)          0         
_________________________________________________________________
dropout_10 (Dropout)         (None, 3, 3, 64)          0         
_________________________________________________________________
flatten_10 (Flatten)         (None, 576)               0         
_________________________________________________________________
dense_19 (Dense)             (None, 3168)              1827936   
_________________________________________________________________
activation_49 (Activation)   (None, 3168)              0         
_________________________________________________________________
dense_20 (Dense)             (None, 10)                31690     
_________________________________________________________________
activation_50 (Activation)   (None, 10)                0         
=================================================================


# after 100 Epoch of training we get the best train acc of 100.00%, the best vall acc of 99.97%
# And we get the acc of in Kaggle test set
