import argparse
import os

import tensorflow as tf

from settings import Settings
from DataHandler import DataHandler
from util import str2bool
import tinynet_architecture as net

def get_model_fn():
    """Creates a model function that builds the net and manages estimator specs."""

    def _small_net_model_fn(features, labels, mode, params):
        """Builds the network model and prepares EstimatorSpecs for prediction, training and evaluation."""
        
        # individual model set-up
        settings = Settings()
        network = net.CIFAR10_NET(settings, features, labels, params)
        logits = network.logits

        # prediction EstimatorSpec
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # training EstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=network.loss, train_op=network.update)

        # evaluation EstimatorSpec
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predicted_classes)
        }
        return tf.estimator.EstimatorSpec(mode, loss=network.loss, eval_metric_ops=eval_metric_ops)

    return _small_net_model_fn

def get_input_fn(mode=None, params=None):
    """Creates an input function that loads the dataset and prepares it for use."""

    def _input_fn(mode=None, params=None):
        """Loads the dataset, decodes, reshapes and preprocesses it for use. Computations performed on CPU."""
        with tf.device('/cpu:0'):
            if mode == 'train':
                dataset = DataHandler(mode, "train", params).prepare_for_train()
            elif mode == 'eval':
                dataset = DataHandler(mode, "validation", params).prepare_for_eval(params.eval_batch_size)
            elif mode == 'test':
                dataset = DataHandler(mode, "test", params).prepare_for_eval(params.eval_batch_size)
            else:
                raise ValueError('_input_fn received invalid MODE')
            return dataset.make_one_shot_iterator().get_next()

    return _input_fn

def build_estimator(run_config, hparams):
    """Builds the estimator object and returns it."""
    return tf.estimator.Estimator(model_fn=get_model_fn(),
           config=run_config,
           params=hparams)

def main(**hparams):

    # Start tensorflow logging
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Using arguments: {}\n'.format(str(hparams)))

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_count={"CPU": hparams['num_cores'], "GPU": hparams['num_gpus']},
        gpu_options=tf.GPUOptions(force_gpu_compatible=True)
    )

    # for-loop for batch jobs
    for i in range(hparams['repeats']):
        tf.logging.info('Commencing iteration {} of {}.'.format((i+1), hparams['repeats']))

        config = tf.estimator.RunConfig(
            model_dir=os.path.join(hparams['output_dir'], '{}-{}'.format(str(hparams['job_id']), str(i+1))),
            tf_random_seed=None,
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            save_checkpoints_secs=None,
            session_config=session_config,
            keep_checkpoint_max=int(hparams['train_steps']/1000)+1,
            keep_checkpoint_every_n_hours=10000,
            log_step_count_steps=100
            #train_distribute=None
        )

        classifier = build_estimator(
            run_config=config,
            hparams=tf.contrib.training.HParams(**hparams)
        )

        # start training and evaluation loop with estimator
        tf.estimator.train_and_evaluate(
            classifier,
            tf.estimator.TrainSpec(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.TRAIN), max_steps=hparams['train_steps']),
            tf.estimator.EvalSpec(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.EVAL), throttle_secs=1, steps=None)
        )

        # compute final test performance
        if hparams['perform_test']:
            classifier.evaluate(input_fn=get_input_fn(mode='test'), name='test')

        tf.logging.info('Finished iteration {} of {}.\n'.format((i+1), hparams['repeats']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data-dir',
        type=str,
        required=True,
        help='The directory where the input data is stored.',
        dest='data_dir')
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='The directory where the output will be stored.',
        dest='output_dir')
    parser.add_argument(
        '-j', '--job-id',
        type=int,
        required=True,
        help='The id this job was assigned during submission, alternatively any unique number distinguish runs.',
        dest='job_id')
    parser.add_argument(
        '-a', '--array-job',
        type=int,
        default=1,
        help='Number of scheduled repeats this job should run for.',
        dest='repeats')
    parser.add_argument(
        '-d', '--logit-dimensions',
        type=int,
        required=True,
        help='Dimension of network logits',
        dest='logit_dims')
    parser.add_argument(
        '-n', '--num-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.',
        dest='num_gpus')
    parser.add_argument(
        '-c', '--num-cpu-cores',
        type=int,
        default=1,
        help='The number of cpu cores available for data preparation.',
        dest='num_cores')
    parser.add_argument(
        '-s', '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.',
        dest='train_steps')
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=128,
        help='Batch size.',
        dest='batch_size')
    parser.add_argument(
        '-e', '--eval-batch-size',
        type=int,
        default=-1,
        help="Evaluation batch size. Defaults to -1, will then use dataset's full evaluation set in one batch",
        dest="eval_batch_size")
    parser.add_argument(
        '-l', '--learning-rate',
        type=float,
        default=5e-4,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.""",
        dest='learning_rate')
    parser.add_argument(
        '-t', '--perform-test',
        type=str2bool,
        default=False,
        help="Whether or not to evaluate test performance after max_steps is reached",
        dest='perform_test')
    parser.add_argument(
        '-g', '--summarise-gradients',
        type=str2bool,
        default=False,
        help="Whether or not to summarise layer weight gradients.",
        dest='sum_grads')
    parser.add_argument(
        '-p', '--preprocess-data',
        type=str2bool,
        default=False,
        help='Whether or not the input data for training should be preprocessed',
        dest='preprocess_data')
    parser.add_argument(
        '-z', '--preprocess-zoom-factor',
        type=float,
        default=1.20,
        help='Zoom factor for pad and crop performed during preprocessing.',
        dest='preprocess_zoom'
    )
    args = parser.parse_args()

    main(**vars(args))
