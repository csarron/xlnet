#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
from scipy.special import softmax
import json
import os
from os.path import join

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import function_builder
import model_utils
from eval import load_examples
from util import flags
from util import logger
from util import tf

# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_string("summary_type", default="first",
                    help="Method used to summarize a sequence "
                         "into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("checkpoint_path", default="",
                    help="the finetuned model path.")
flags.DEFINE_string("train_file", default="",
                    help="Path of train file.")
flags.DEFINE_string("eval_file", default="",
                    help="Path of prediction file.")
flags.DEFINE_string("eval_example_file", default="",
                    help="Path of prediction file.")
# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
                          "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

# training
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_steps", default=1000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=8,
                     help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# evaluation
flags.DEFINE_bool("do_predict", default=False, help="whether to do prediction")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=128,
                     help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128,
                     help="batch size for prediction.")
# task specific
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")
flags.DEFINE_bool("uncased", default=False,
                  help="Use uncased.")
flags.DEFINE_bool("is_regression", default=False,
                  help="Whether it's a regression task.")
flags.DEFINE_integer("num_classes", default=2,
                     help="batch size for prediction")
flags.DEFINE_string("task", default="boolq",
                    help="boolq or race or qqp or mnli")
flags.DEFINE_string("tune_scopes", default="",
                    help="fine tune scopes")
flags.DEFINE_string("init_scopes", default="",
                    help="fine initialize variable scopes")
flags.DEFINE_bool("decompose", default=False, help="whether to use decompose")
flags.DEFINE_integer("sep_layer", default=9,
                     help="which layer to start decompose")
flags.DEFINE_integer("max_first_length", default=25,
                     help="max_first_length")
flags.DEFINE_bool("supervise", default=False,
                  help="whether to use distill")
flags.DEFINE_float("dl_alpha", default=0.5, help="distill loss hyper param")
flags.DEFINE_float("ul_beta", default=0.5, help="upper layer loss hyper param")
flags.DEFINE_float("ll_gamma", default=1.0, help="label loss hyper param")
flags.DEFINE_float("temperature", default=2.0, help="distill temperature")
FLAGS = flags.FLAGS


def file_based_input_fn_builder(FLAGS, is_training):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    input_file = FLAGS.train_file if is_training else FLAGS.eval_file
    if FLAGS.decompose:
        max_f_length = FLAGS.max_first_length + 2
        max_s_length = FLAGS.max_seq_length - max_f_length
        name_to_features = {
            "feature_id": tf.io.FixedLenFeature([], tf.int64),
            "seq1_ids": tf.io.FixedLenFeature([max_f_length], tf.int64),
            "seq2_ids": tf.io.FixedLenFeature([max_s_length], tf.int64),
        }
    else:
        seq_length = FLAGS.max_seq_length
        name_to_features = {
            "feature_id": tf.FixedLenFeature([], tf.int64),
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        }
    if is_training:
        name_to_features["cls"] = tf.FixedLenFeature([], tf.int64)
    logger.info("Input tfrecord file {}".format(input_file))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params, input_context=None):
        """The actual input function."""
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        elif FLAGS.do_eval:
            batch_size = FLAGS.eval_batch_size
        else:
            batch_size = FLAGS.predict_batch_size

        d = tf.data.TFRecordDataset(input_file)
        # Shard the dataset to difference devices
        if input_context is not None:
            logger.info("Input pipeline id %d out of %d",
                        input_context.input_pipeline_id,
                        input_context.num_replicas_in_sync)
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
            d = d.repeat()

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=is_training))

        return d

    return input_fn


def get_model_fn(FLAGS):
    def model_fn(features, labels, mode, params):
        # ### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        return_dict = function_builder.get_classification_loss(
            FLAGS, features, is_training)
        # per_example_loss = return_dict["per_example_loss"]
        cls_logits = return_dict["cls_logits"]
        # ### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        logger.info('#params: {}'.format(num_params))

        # ### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # label_ids = tf.reshape(features["cls"], [-1])
            predictions = {
                "cls_logits": cls_logits,
                # "cls": label_ids,
            }

            if FLAGS.use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)
            return output_spec

        def compute_loss(log_probs, positions, depth):
            one_hot_positions = tf.one_hot(
                positions, depth=depth, dtype=tf.float32)

            loss = - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
            loss = tf.reduce_mean(loss)
            return loss

        cls_log_probs = return_dict["cls_log_probs"]
        total_loss = compute_loss(cls_log_probs, features["cls"],
                                  depth=FLAGS.num_classes)

        # ### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        monitor_dict = {'loss/cls': total_loss,
                        "lr": learning_rate}

        # ### Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            # ### Creating host calls
            if not FLAGS.is_regression:
                label_ids = tf.reshape(features['cls'], [-1])
                predictions = tf.argmax(cls_logits, axis=-1,
                                        output_type=label_ids.dtype)
                is_correct = tf.equal(predictions, label_ids)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                monitor_dict["accuracy"] = accuracy

                host_call = function_builder.construct_scalar_host_call(
                    monitor_dict=monitor_dict,
                    model_dir=FLAGS.model_dir,
                    prefix="train/",
                    reduce_fn=tf.reduce_mean)
            else:
                host_call = None

            train_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op,
                host_call=host_call,
                scaffold_fn=scaffold_fn)
        else:
            train_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn


def main(_):

    # ### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval, `do_predict` or "
            "`do_submit` must be True.")

    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn(FLAGS)
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if FLAGS.do_train:
        train_input_fn = file_based_input_fn_builder(FLAGS, is_training=True)

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_predict:

        pred_input_fn = file_based_input_fn_builder(FLAGS, is_training=False)
        cur_results = []
        checkpoint_path = FLAGS.checkpoint_path
        if not checkpoint_path:
            checkpoint_path = None
        for result in estimator.predict(input_fn=pred_input_fn,
                                        checkpoint_path=checkpoint_path,
                                        yield_single_examples=True):

            if len(cur_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(cur_results)))

            unique_id = int(result["feature_id"])
            cls_logits = (
                [float(x) for x in result["cls_logits"].flat])
            cur_results.append((unique_id, cls_logits))
        eval_examples = load_examples(FLAGS.eval_example_file)
        final_predictions = collections.OrderedDict()
        labels, predicted_labels = [], []
        for cur_result in cur_results:
            unique_id, cls_logits = cur_result
            item = eval_examples[int(unique_id)]
            orig_id = item['orig_id']
            scores = softmax(cls_logits)
            if 'label' in item:
                answer_cls = item['label']['cls']
                label_id = int(answer_cls)
                labels.append(label_id)
            predicted_label = int(np.argmax(scores))
            final_predictions[orig_id] = predicted_label
            predicted_labels.append(predicted_label)

        acc = accuracy_score(y_true=labels, y_pred=predicted_labels)
        acc *= 100.0
        f1_str = ''
        if FLAGS.num_classes == 2:
            f1 = f1_score(y_true=labels, y_pred=predicted_labels)
            f1 *= 100.0
            f1_str = ", f1={:.4f}".format(f1)
        ckpt = os.path.basename(checkpoint_path) if checkpoint_path else ''
        dec_suffix = '_s{}'.format(FLAGS.sep_layer) if FLAGS.decompose else ''
        prediction_prefix = FLAGS.eval_file + ckpt + dec_suffix + '.predictions'
        prediction_file = prediction_prefix + '.json'
        pred_path = FLAGS.prediction_file or prediction_file
        pred_dir = os.path.dirname(pred_path)
        if not tf.io.gfile.exists(pred_dir):
            tf.io.gfile.makedirs(pred_dir)
        final_predictions_data = final_predictions
        with tf.io.gfile.GFile(pred_path, "w") as f:
            f.write(json.dumps(final_predictions_data, indent=2,
                               ensure_ascii=False) + "\n")
        logger.info("final predictions written to {}".format(pred_path))
        logger.info("acc={:.4f}{}".format(acc, f1_str))


if __name__ == "__main__":
    tf.app.run()
