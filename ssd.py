"""Keras implementation of SSD."""

import keras.backend as K
import keras
# from keras.layers import Activation
# from keras.layers import AtrousConvolution2D
# from keras.layers import Convolution2D
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import GlobalAveragePooling2D
# from keras.layers import Input
# from keras.layers import MaxPooling2D
# from keras.layers import concatenate
# from keras.layers import Reshape
# from keras.layers import ZeroPadding2D
# from keras.models import Model

from ssd_layers import Normalize
from ssd_layers import PriorBox


def SSD300(input_shape, num_classes=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """

    inputs = keras.layers.Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    
    def get_name(tensor):
        s = tensor.name
        try:
            i = s.index('/')
            s = s[:i]
        except ValueError:
            pass
        return s

    def conv2d(filters, kernel_size=(3, 3), name=None, **kwargs):
        default_kwargs = {
            'activation': 'relu',
            'padding': 'same',
        }
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        return keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            #strides=(1, 1),
            name=name,
            **kwargs
        )

    def maxpool(name, pool_size=(2, 2), strides=(2, 2)):
        return keras.layers.MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            padding='same',
            name=name,
        )

    mbox_loc_tensors = []
    mbox_conf_tensors = []
    mbox_priorbox_tensors = []

    def add_mbox_loc(input_tensor, num_priors, dense=False):
        x = input_tensor
        if dense:
            name = get_name(input_tensor) + '_mbox_loc_flat'
            x = keras.layers.Dense(num_priors * 4, name=name)(x)
        else:
            name = get_name(input_tensor) + '_mbox_loc'
            x = conv2d(num_priors * 4, (3, 3), name)(x)
            x = keras.layers.Flatten(name = name + '_flat')(x)
        mbox_loc_tensors.append(x)

    def add_mbox_conf(input_tensor, num_priors, num_classes=num_classes,
            dense=False):
        name_suffix = ''
        if num_classes != 21:
            name_suffix = '_{}'.format(num_classes)
        x = input_tensor
        if dense:
            name = get_name(input_tensor) + '_mbox_conf_flat'
            x = keras.layers.Dense(num_priors * num_classes, name = name + name_suffix)(x)
        else:
            name = get_name(input_tensor) + '_mbox_conf'
            x = conv2d(num_priors * num_classes, (3, 3), name + name_suffix)(x)
            x = keras.layers.Flatten(name = name + '_flat')(x)
        mbox_conf_tensors.append(x)

    def add_mbox_priorbox(input_tensor, num_priors,
            img_size=img_size,
            min_size=None, max_size=None, aspect_ratios=None,
            variances=[0.1],
            reshape=False):
        x = input_tensor
        if reshape:
            name = get_name(input_tensor) + '_reshaped'
            x = keras.layers.Reshape(target_shape, name=name)(x)
        name = get_name(input_tensor) + '_mbox_priorbox'
        x = PriorBox(
            img_size, min_size, max_size,
            aspect_ratios=aspect_ratios,
            variances=variances,
            name=name,
        )(x)
        mbox_priorbox_tensors.append(x)

    ## -------- Input --------
    x = inputs
    ## -------- Block 1 --------
    x = conv2d(64, (3, 3), 'conv1_1')(x)
    x = conv2d(64, (3, 3), 'conv1_2')(x)
    x = maxpool('pool1')(x)
    ## -------- Block 2 --------
    x = conv2d(128, (3, 3), 'conv2_1')(x)
    x = conv2d(128, (3, 3), 'conv2_2')(x)
    x = maxpool('pool2')(x)
    ## -------- Block 3 --------
    x = conv2d(256, (3, 3), 'conv3_1')(x)
    x = conv2d(256, (3, 3), 'conv3_2')(x)
    x = maxpool('pool3')(x)
    ## -------- Block 4 --------
    x = conv2d(512, (3, 3), 'conv4_1')(x)
    x = conv2d(512, (3, 3), 'conv4_2')(x)
    x = conv2d(512, (3, 3), 'conv4_3')(x)
    conv4_3 = x
    x = maxpool('pool4')(x)
    ## -------- Block 5 --------
    x = conv2d(512, (3, 3), 'conv5_1')(x)
    x = conv2d(512, (3, 3), 'conv5_2')(x)
    x = conv2d(512, (3, 3), 'conv5_3')(x)
    x = maxpool('pool5', pool_size=(3, 3), strides=(1, 1))(x)
    ## -------- FC6 --------
    x = conv2d(1024, (3, 3), 'fc6', dilation_rate=(6, 6))(x)
    # x = keras.layers.Dropout(0.5, name='drop6')(x)
    ## -------- FC7 --------
    x = conv2d(1024, (1, 1), 'fc7')(x)
    fc7 = x
    # x = keras.layers.Dropout(0.5, name='drop7')(x)
    ## -------- Block 6 --------
    x = conv2d(256, (1, 1), 'conv6_1')(x)
    x = conv2d(512, (3, 3), 'conv6_2', strides=(2, 2))(x)
    conv6_2 = x
    ## -------- Block 7 --------
    x = conv2d(128, (1, 1), 'conv7_1')(x)
    x = keras.layers.ZeroPadding2D()(x)
    x = conv2d(256, (3, 3), 'conv7_2', strides=(2, 2), padding='valid')(x)
    conv7_2 = x
    ## -------- Block 8 --------
    x = conv2d(128, (1, 1), 'conv8_1')(x)
    x = conv2d(256, (3, 3), 'conv8_2', strides=(2, 2))(x)
    conv8_2 = x
    ## -------- Last Pool --------
    x = keras.layers.GlobalAveragePooling2D(name='pool6')(x)
    pool6 = x

    ## -------- Predictions --------
    mbox_loc_tensors = []
    mbox_conf_tensors = []
    mbox_priorbox_tensors = []
    # --------
    x = Normalize(20, name='conv4_3_norm')(conv4_3)
    num_priors = 3
    add_mbox_loc(x, num_priors)
    add_mbox_conf(x, num_priors, num_classes)
    add_mbox_priorbox(
        x, num_priors,
        min_size=30.0, max_size=None, aspect_ratios=[2],
        variances=[0.1, 0.1, 0.2, 0.2],
    )
    # --------
    min_size = 60.0
    for x in (fc7, conv6_2, conv7_2, conv8_2):
        num_priors = 6
        add_mbox_loc(x, num_priors)
        add_mbox_conf(x, num_priors)
        add_mbox_priorbox(
            x, num_priors,
            min_size=min_size, max_size=min_size + 54.0, aspect_ratios=[2, 3],
            variances=[0.1, 0.1, 0.2, 0.2],
        )
        min_size += 54.0
    # --------
    x = pool6
    num_priors = 6
    add_mbox_loc(x, num_priors, dense=True)
    add_mbox_conf(x, num_priors, dense=True)
    add_mbox_priorbox(
        x, num_priors,
        min_size=276.0, max_size=330.0, aspect_ratios=[2, 3],
        variances=[0.1, 0.1, 0.2, 0.2],
        reshape=True,
    )

    # Gather all predictions
    mbox_loc = keras.layers.concatenate(mbox_loc_tensors, axis=1, name='mbox_loc')
    mbox_conf = keras.layers.concatenate(mbox_conf_tensors, axis=1, name='mbox_conf')
    mbox_priorbox = keras.layers.concatenate(mbox_priorbox_tensors, axis=1, name='mbox_priorbox')

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = keras.layers.Reshape(
        (num_boxes, 4),
        name='mbox_loc_final')(mbox_loc)
    mbox_conf = keras.layers.Reshape(
        (num_boxes, num_classes),
        name='mbox_conf_logits')(mbox_conf)
    mbox_conf = keras.layers.Activation(
        'softmax',
        name='mbox_conf_final')(mbox_conf)

    predictions = keras.layers.concatenate(
        [mbox_loc, mbox_conf, mbox_priorbox],
        axis=2,
        name='predictions')

    model = keras.models.Model(inputs, predictions)

    return model
