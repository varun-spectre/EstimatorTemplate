
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

#preprocess your data
#initializing random values similar to mnist
x_train = np.random.randn(60000,28,28,1)
y_train = np.random.randint(10,size=(60000,1))
x_test = np.random.randn(10000,28,28,1)
y_test = np.random.randint(10, size=(10000,1))


#create your own model

def create_model():
    data_format = 'channels_last'
    input_shape= [-1,28,28,1]
    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)

    return tf.keras.Sequential(
      [
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])


# Returns Estimator spec object
# Dont have to set mode manually
# Add other parameters if needed in logging hook
# use custom learning rate and decay base

def my_model(features, labels, mode):
    labels= tf.cast(labels,tf.int64)
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    model = create_model()
    Ylogits = model(input_layer, training=True)
    if mode == tf.estimator.ModeKeys.PREDICT:
        Ylogits = model(input_layer, training=False)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=Ylogits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(Ylogits, name="softmax_tensor")
      }
    
    
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=Ylogits)
    accuracy =  tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    # Configure the Training Op (for TRAIN mode)
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy[1]}, every_n_iter=10)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
      }
    starter_learning_rate = 0.001
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(starter_learning_rate, tf.train.get_global_step(), 2000, 0.96, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,training_hooks = [logging_hook])
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


#train_steps depends on how big is your data(General value - 10^5)
#eval_steps depends on your test data size and batch size(floor(test_size/batch_size))
#shuffle must be false in eval_input_fn
def train(run_config):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_steps = 10000
    eval_steps = 7   
    classifier = tf.estimator.Estimator(model_fn=my_model,model_dir="where/you/want/to/save",config = run_config)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},y=y_train,batch_size=100,num_epochs=None,shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_test},y=y_test,shuffle=False)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=eval_steps, throttle_secs=120)
    return tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


#runconfig help us fix number of max checkpoints we want to save, log step count ect;
def main():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.estimator.RunConfig(session_config=sess_config,model_dir="where/you/want/to/save", keep_checkpoint_max=200,
                                    log_step_count_steps=10, save_checkpoints_steps =200)
    train(config)


main()


#use tensorboard -logdir "wherr you have saved" to view on tensorboard

#prediction for your own file

pred_file = 'file/you/want/to/test'
#if any preprocessing done, do the same
file = np.reshape(pred_file,(28,28,1))
file = np.reshape(file,(1,28,28,1))
#if you dont specify anything your model will take last saved checkpoint
#if you want to your model to make prediction on specific checkpoint, edit the checkpoint file in the saved folder 
def pred_model():
    emot_classifier = tf.estimator.Estimator(model_fn = my_model,model_dir = 'where/you/have/saved')
    tensors_to_log = {"probabilties":"softmax_tensor"}
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':pred_file},shuffle= False)
    pred_results = classifier.predict(input_fn = pred_input_fn)
    print(pred_results)

pred_model()

