import os
# import shutil
import argparse
# import glob
# import imageio
from dataset import read_font_data, FontDataManager
from utils import render_fonts_image

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def leaky_relu(x, alpha=0.2):
    # TODO: is this memory efficient?
    return tf.maximum(x, x * alpha)

def cnn_model_fn(features, targets, mode, params):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 80, 80, 1])

    conv1_64_64 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[64,64],
        padding="same",
        activation=leaky_relu)
    conv2_64_64 = tf.layers.conv2d(
        inputs=conv1_64_64,
        filters=8,
        kernel_size=[64,64],
        padding="same",
        activation=leaky_relu)
        
    conv1_32_32 = tf.layers.conv2d(
        inputs=conv2_64_64,
        filters=32,
        kernel_size=[32,32],
        padding="same",
        activation=leaky_relu)
    
    conv2_32_32 = tf.layers.conv2d(
        inputs=conv1_32_32,
        filters=32,
        kernel_size=[32,32],
        padding="same",
        activation=leaky_relu)
        
    conv1_16_16 = tf.layers.conv2d(
        inputs=conv2_32_32,
        filters=64,
        kernel_size=[16,16],
        padding="same",
        activation=leaky_relu)
    
    conv2_16_16 = tf.layers.conv2d(
        inputs=conv1_16_16,
        filters=64,
        kernel_size=[16,16],
        padding="same",
        activation=leaky_relu)
        
    conv1_7_7 = tf.layers.conv2d(
        inputs=conv2_16_16,
        filters=128,
        kernel_size=[7,7],
        padding="same",
        activation=leaky_relu)
    
    conv2_7_7 = tf.layers.conv2d(
        inputs=conv1_7_7,
        filters=128,
        kernel_size=[7,7],
        padding="same",
        activation=leaky_relu)
    
    conv1_3_3 = tf.layers.conv2d(
        inputs=conv2_7_7,
        filters=128,
        kernel_size=[3,3],
        padding="same",
        activation=leaky_relu)

    conv2_3_3 = tf.layers.conv2d(
        inputs=conv1_3_3,
        filters=1,
        kernel_size=[3,3],
        padding="same",
        activation=leaky_relu)    

    pool1 = tf.layers.max_pooling2d(inputs=conv2_3_3, pool_size=[2, 2], strides=2)
    dropout = tf.layers.dropout(inputs=pool1, rate=params["dropout_rate"], training=mode == learn.ModeKeys.TRAIN)
    output = tf.sigmoid(tf.reshape(dropout, [-1, 40, 40]))
    loss = tf.reduce_mean(tf.abs(output - targets))

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="Adam")

    predictions = {"image": output, "loss": tf.identity(loss, name="loss")}

    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def main(unused_argv):
    source_font = FLAGS.source_font
    target_font = FLAGS.target_font
    num_examples = FLAGS.num_examples
    num_validation = FLAGS.num_validations
    split = num_examples - num_validation
    train_keep_prob = FLAGS.keep_prob
    num_iter = FLAGS.iter
    frame_dir = FLAGS.frame_dir
    checkpoint_steps = FLAGS.ckpt_steps
    num_checkpoints = FLAGS.num_ckpt
    checkpoints_dir = FLAGS.ckpt_dir

    dataset = FontDataManager(source_font, target_font, num_examples, split)

    train_x, train_y = dataset.get_train()
    validation_x, validation_y = dataset.get_validation()
    # print(train_x.shape)
    # print(train_y.shape)

    # Create the Estimator
    model_params = {"dropout_rate": 1-train_keep_prob}
    model = learn.Estimator(model_fn=cnn_model_fn, params=model_params)    

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    model.fit(
        x=train_x,
        y=train_y,
        batch_size=20,
        steps=num_iter,
        monitors=[logging_hook])

    # metrics = {
    #   "accuracy":
    #       learn.MetricSpec(
    #           metric_fn=tf.metrics.mean_absolute_error, prediction_key="image"),
    # }
    
    # eval_results = model.evaluate(x=validation_x, y=validation_y, metrics=metrics)
    # print(eval_results)

    predictions = model.predict(x=validation_x, outputs="image")
    images = np.array(map(lambda x: x["image"], list(predictions)))
    render_fonts_image(images, os.path.join(frame_dir, "test.png"), 10)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--mode', type=str, default='train',
                        help='could be either infer or train')
    parser.add_argument('--model', type=str, default='medium',
                        help='type of model, could small, medium or big')
    parser.add_argument('--source_font', type=str, default=None,
                        help='npy bitmap for the source font')
    parser.add_argument('--target_font', type=str, default=None,
                        help='npy bitmap for the target font')
    parser.add_argument('--num_examples', type=int, default=2000,
                        help='number of examples for training')
    parser.add_argument('--num_validations', type=int, default=50,
                        help='number of chars for validation')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default to 0.01')
    parser.add_argument('--keep_prob', type=float, default=0.9,
                        help='keep probability for dropout layer, defaults to 0.9')
    parser.add_argument('--iter', type=int, default=1000,
                        help='number of iterations')
    parser.add_argument('--tv', type=float, default=0.0002,
                        help='weight for tv loss, use to force smooth output')
    parser.add_argument('--alpha', type=float, default=-1.0,
                        help='alpha slope for leaky relu if non-negative, otherwise use relu')
    parser.add_argument('--ckpt_steps', type=int, default=50,
                        help='number of steps between two checkpoints')
    parser.add_argument('--num_ckpt', type=int, default=5,
                        help='number of model checkpoints to keep')
    parser.add_argument('--ckpt_dir', type=str, default='/tmp/checkpoints',
                        help='directory for store checkpoints')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint file path to restore for inference')
    parser.add_argument('--capture_frame', type=bool, default=True,
                        help='capture font images between iterations and compiled to gif')
    parser.add_argument('--frame_dir', type=str, default='/tmp/frames',
                        help='temporary directory to store font image frames')
    parser.add_argument('--summary_dir', type=str, default='/tmp/summary',
                        help='directory for storing data')
    parser.add_argument('--bitmap_dir', type=str, default='/tmp/bitmap',
                        help='directory for saving inferred bitmap')
    FLAGS = parser.parse_args()

    tf.app.run()
