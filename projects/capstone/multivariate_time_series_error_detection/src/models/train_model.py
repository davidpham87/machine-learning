from tensorflow import keras
kl = keras.layers


def add_common_layers(y):
    y = kl.BatchNormalization()(y)
    y = kl.LeakyReLU()(y)
    return y

def grouped_convolution(y, nb_channels, _strides, cardinality=4):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return kl.Conv1D(nb_channels, kernel_size=10, strides=_strides, padding='same')(y)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = kl.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(kl.Conv1(_d, kernel_size=10, strides=_strides, padding='same')(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = kl.concatenate(groups)

    return y

def residual_block(y, nb_channels_in, nb_channels_out, cardinality=4, _strides=1, _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = kl.Conv1D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = add_common_layers(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)

    y = kl.Conv1D(nb_channels_out, kernel_size=1, strides=1, padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = kl.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != 1:
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = kl.Conv1D(nb_channels_out, kernel_size=1, strides=_strides, padding='same')(shortcut)
        shortcut = kl.BatchNormalization()(shortcut)

    y = kl.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = kl.LeakyReLU()(y)

    return y
