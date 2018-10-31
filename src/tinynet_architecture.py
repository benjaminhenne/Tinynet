import tensorflow as tf
from activation_functions import identity_activation, swish

class CIFAR10_NET(object):
    """

    """

    def __init__(self, Settings, features, labels, hparams):
        """

        """
        self.settings = Settings
        self.hparams = hparams
        print(features)
        #Variables
        self.X = features
        self.y = labels
        self.learning_rate = hparams.learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        #NETWORK
        if (self.settings.network_structure == "ianntf"):
            self.logits, layer_names = self.build_ianntf_network()
        elif self.settings.network_structure == "ianntf_fixed":
            self.logits, layer_names = self.build_ianntf_network_fixed(namescope="simple_net")
        else:
            self.logits, layer_names = self.build_network(namescope="simple_net")

        #Objective
        self.penalty = tf.constant(0)
        with tf.name_scope('objective'):
            self.xentropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

            if self.settings.l1_regularize and self.settings.l2_regularize:
                raise Exception("[FATAL] L1 Regularization AND L2 regularization is not possible")
            elif self.settings.l1_regularize:
                weight_sets = []
                for layer in layer_names:
                    weight_sets.append([v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if layer+'/act_weight' in v.name])

                l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self.settings.l1_regularizer_scale, scope="l1_regularization")
                penalties = [tf.contrib.layers.apply_regularization(l1_regularizer, weights) for weights in weight_sets]
                self.penalty = tf.add_n(penalties)
                self.loss = self.xentropy + self.penalty

                tf.summary.scalar('penalty', self.penalty)
            elif self.settings.l2_regularize:
                self.l2 = [tf.nn.l2_loss(v) for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'weight' and not 'act_weight' in v.name]
                self.weights_norm = tf.reduce_sum(input_tensor=self.settings.l2_lambda*tf.stack(self.l2), name='weights_norm')
                self.loss = self.xentropy + self.weights_norm
            else:
                self.loss = self.xentropy

            #accuracy node
            self.accuracy = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('xentropy', self.xentropy)

        #Optimization
        with tf.name_scope('optimization'):
            if self.settings.optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            self.minimize = self.optimizer.minimize(self.loss)
            varlist = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=varlist)
            self.update = self.optimizer.apply_gradients(grads_and_vars=self.gradients, global_step=tf.train.get_global_step())

        #summaries
        self.summaries = tf.summary.merge_all()


    def build_network(self, namescope=None):
        """

        """
        with tf.name_scope(namescope):

            state = self.conv_layer_mulit_act(layer_input=self.X, filter_shape=[5,5,3,64], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope=namescope+'/conv1')
            state = tf.nn.dropout(state, keep_prob=0.5, name=namescope+'/dropout1')
            state = self.conv_layer_mulit_act(layer_input=state, filter_shape=[3,3,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope=namescope+'/conv2')

            state = tf.nn.max_pool(state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')

            state = self.conv_layer_mulit_act(layer_input=state, filter_shape=[1,1,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope=namescope+'/conv3')
            state = tf.nn.dropout(state, keep_prob=0.5, name=namescope+'/dropout3')
            state = self.conv_layer_mulit_act(layer_input=state, filter_shape=[5,5,64,64], strides=[1,1,1,1], padding='SAME', AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope=namescope+'/conv4')

            state = tf.nn.max_pool(state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool4')

            state = self.dense_multi_act_layer(layer_input=state, W_shape=[384], AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope=namescope+'/dense5')
            state = tf.nn.dropout(state, keep_prob=0.5, name=namescope+'/dropout5')
            state = self.dense_multi_act_layer(layer_input=state, W_shape=[192], AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope=namescope+'/dense6')
            # output layer
            logits = self.dense_multi_act_layer(layer_input=state, W_shape=[self.hparams.logit_dims], AF_set=None, af_weights_init=None, varscope=namescope+'/denseout')
            return logits, ["conv1", "conv2", "conv3", "conv4", "dense5", "dense6"]

    def build_ianntf_network(self, namescope=None):
        """

        """
        with tf.name_scope(namescope):

            state = self.conv_layer_mulit_act(layer_input=self.X, filter_shape=[5,5,3,16], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope='conv1')

            state = tf.nn.max_pool(state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')

            state = self.conv_layer_mulit_act(layer_input=state, filter_shape=[3,3,16,32], strides=[1,1,1,1], padding='SAME', AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope='conv2')
            state = tf.nn.max_pool(state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool4')

            state = self.dense_multi_act_layer(layer_input=state, W_shape=[512], AF_set=self.settings.af_set, af_weights_init=self.settings.af_inits, varscope='dense3')

            logits = self.dense_multi_act_layer(layer_input=state, W_shape=[self.hparams.logit_dims], AF_set=None, af_weights_init=None, varscope='denseout')
            return logits, ["conv1", "conv2", "dense3"]


    def build_ianntf_network_fixed(self, namescope=None):
        """

        """
        with tf.name_scope(namescope):

            state = self.conv_layer_mulit_act(layer_input=self.X, filter_shape=[5,5,3,16], strides=[1,1,1,1], padding='SAME', bias_init=0.0, AF_set=None, af_weights_init=self.settings.af_inits, varscope=namescope+'/conv1')

            state = tf.nn.max_pool(state, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool2')

            state = self.conv_layer_mulit_act(layer_input=state, filter_shape=[3,3,16,32], strides=[1,1,1,1], padding='SAME', AF_set=None, af_weights_init=self.settings.af_inits, varscope=namescope+'/conv2')
            state = tf.nn.max_pool(state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=namescope+'/pool4')

            state = self.dense_multi_act_layer(layer_input=state, W_shape=[512], AF=None, AF_set=None, af_weights_init=self.settings.af_inits, varscope=namescope+'/dense3')

            beta = tf.get_variable('act_bias/swish', initializer = 0.1)
            state = swish(state, beta)

            logits = self.dense_multi_act_layer(layer_input=state, W_shape=[self.hparams.logit_dims], AF_set=None, af_weights_init=None, varscope=namescope+'/denseout')
            return logits, ["conv1", "conv2", "dense3"]


    def dense_multi_act_layer(self, layer_input, W_shape, b_shape=[-1], bias_init=0.1, AF=None, AF_set=None, af_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted', swish_beta_trainable=True, preblend_batchnorm=False, reuse=False, varscope=None):
        """

        """
        with tf.variable_scope(varscope, reuse=reuse):
            flat_input = tf.layers.flatten(layer_input)
            input_dims = flat_input.get_shape().as_list()[1]
            W_shape = [input_dims, W_shape[0]]
            if b_shape == [-1]:
                b_shape = [W_shape[-1]]
            W = tf.get_variable('weights', W_shape, initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2./(W_shape[0]*W_shape[1])))) # stddev=0.1
            b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
            state = tf.matmul(flat_input, W)
            if b_shape != [0]:
                state += b
            if not AF is None:
                return AF(state)
            if AF_set is None:
                return state
            return self.activate(state, AF_set, af_weights_init, varscope)

    def conv_layer_mulit_act(self, layer_input, filter_shape, b_shape=[-1], strides=[1,1,1,1], padding='SAME', bias_init=0.1, AF=None, AF_set=None, af_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted', swish_beta_trainable=True, preblend_batchnorm=False, reuse=False, varscope=None):
        """

        """
        with tf.variable_scope(varscope, reuse=reuse):
            if b_shape == [-1]:
                b_shape = [filter_shape[-1]]
            filter_initializer = tf.truncated_normal_initializer(stddev=tf.sqrt(2./(filter_shape[0]*filter_shape[1]*filter_shape[2]))) # stddev=0.1
            filter = tf.get_variable('filter', filter_shape, initializer=filter_initializer)
            b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
            state = tf.nn.conv2d(layer_input, filter, strides, padding)
            if b_shape != [0]:
                state += b
            if not AF is None:
                return AF(state)
            if AF_set is None:
                return state
            return self.activate(state, AF_set, af_weights_init, varscope)

    def activate(self, drive, af_set, af_weights_init, varscope):
        """

        """
        activations = []
        shape = drive.get_shape().as_list()
        for af, af_name in af_set:
            weight = tf.get_variable('act_weight/'+af_name, initializer=af_weights_init[af_name](shape))
            if af_name == "swish":
                #print(af_inits[af_name](shape))
                beta = tf.get_variable('act_bias/'+af_name, initializer = af_weights_init[af_name](shape))
                tf.summary.scalar('act_bias/swish_beta', beta)
                activations.append(weight * af(drive, beta))
                #tf.summary.scalar('act_weight/'+name+"/test"+af_name, activations[-1][0])
            else:
                activations.append(weight * af(drive))

            tf.summary.scalar('act_weight/'+af_name, weight)

        return tf.add_n(activations)
