import tensorflow as tf

def identity_activation(drive, name=None):
    return drive
    
def swish(drive, beta, name=None):
    return drive * tf.nn.sigmoid(drive * beta, name=name)


