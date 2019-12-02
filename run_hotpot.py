# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import logging

import numpy as np
import six
import tensorflow as tf
from absl import flags

import function_builder
import model_utils
import xlnet
from eval import get_em_f1
from eval import load_examples

if six.PY2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger('xlnet')

logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(levelname)s:%(asctime)s.%(msecs)03d:"
                        "%(pathname)s:%(lineno)d: %(message)s",
                        "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.propagate = False

# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length.")
flags.DEFINE_string("summary_type", default="last",
                    help="Method used to summarize a sequence into a vector.")
flags.DEFINE_bool("use_bfloat16", default=False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_bool("init_global_vars", default=False,
                  help="If true, init all global vars. If false, init "
                       "trainable vars only.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("train_record_file", default="",
                    help="Path of train file.")
flags.DEFINE_string("eval_record_file", default="",
                    help="Path of prediction file.")
flags.DEFINE_string("eval_example_file", default="",
                    help="Path of prediction file.")

# Data preprocessing config
flags.DEFINE_integer("max_seq_length",
                     default=512, help="Max sequence length")
flags.DEFINE_integer("max_query_length",
                     default=64, help="Max query length")
flags.DEFINE_integer("doc_stride",
                     default=128, help="Doc stride")
flags.DEFINE_integer("max_answer_length",
                     default=64, help="Max answer length")
flags.DEFINE_bool("uncased", default=False, help="Use uncased data.")

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

# Training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_integer("train_batch_size", default=48,
                     help="batch size for training")
flags.DEFINE_integer("train_steps", default=8000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("max_save", default=5,
                     help="Max number of checkpoints to save. "
                          "Use 0 to save all.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")

# Optimization
flags.DEFINE_float("learning_rate", default=3e-5, help="initial learning rate")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")
flags.DEFINE_float("lr_layer_decay_rate", default=0.75,
                   help="Top layer: lr[L] = FLAGS.learning_rate."
                        "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")

# Eval / Prediction
flags.DEFINE_bool("do_predict", default=False, help="whether to do predict")
flags.DEFINE_integer("predict_batch_size", default=32,
                     help="batch size for prediction")
flags.DEFINE_integer("n_best_size", default=5,
                     help="n best size for predictions")
flags.DEFINE_integer("start_n_top", default=5, help="Beam size for span start.")
flags.DEFINE_integer("end_n_top", default=5, help="Beam size for span end.")
flags.DEFINE_string("target_eval_key", default="best_f1",
                    help="Use has_ans_f1 for Model I.")

FLAGS = flags.FLAGS


def input_fn_builder(input_file, seq_length, is_training, drop_remainder,
                     num_threads=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "feature_id": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["answer_start"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["answer_end"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["cls"] = tf.FixedLenFeature([], tf.int64)

    logger.info("Input tfrecord file: {}".format(input_file))

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

    def input_fn(params):
        """The actual input function."""
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        else:
            batch_size = FLAGS.predict_batch_size

        d = tf.data.TFRecordDataset(input_file)
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
            d = d.repeat()
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_threads,
                drop_remainder=drop_remainder))
        d = d.prefetch(1024)

        return d

    return input_fn


def get_hotpot_qa_outputs(FLAGS, features, is_training):
    """Loss for downstream span-extraction QA tasks such as SQuAD."""

    input_ids = features["input_ids"]
    seg_id = features["segment_ids"]
    input_mask_int = tf.cast(tf.cast(input_ids, tf.bool), tf.int32)
    cls_index = tf.reshape(tf.reduce_sum(input_mask_int, axis=1), [-1])
    p_mask = tf.cast(tf.cast(seg_id, tf.bool), tf.float32)
    # inp_mask = tf.transpose(features["input_mask"], [1, 0])
    # cls_index = tf.reshape(features["cls_index"], [-1])
    input_ids = tf.transpose(input_ids, [1, 0])
    input_mask = 1 - tf.cast(input_mask_int, tf.float32)
    input_mask = tf.transpose(input_mask, [1, 0])
    seg_id = tf.transpose(seg_id, [1, 0])
    seq_len = tf.shape(input_ids)[0]

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=input_ids,
        seg_ids=seg_id,
        input_mask=input_mask)
    output = xlnet_model.get_sequence_output()
    initializer = xlnet_model.get_initializer()

    return_dict = {}
    with tf.variable_scope("logits"):
        # logits: seq, batch_size, 2
        logits = tf.layers.dense(output, 2, kernel_initializer=initializer)

        # logits: 2, batch_size, seq
        logits = tf.transpose(logits, [2, 1, 0])

        # start_logits: batch_size, seq
        # end_logits: batch_size, seq
        start_logits, end_logits = tf.unstack(logits, axis=0)

        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)

        end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
        end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)

    if is_training:
        return_dict["start_log_probs"] = start_log_probs
        return_dict["end_log_probs"] = end_log_probs
    else:
        return_dict["start_logits"] = start_logits
        return_dict["end_logits"] = end_logits

    # an additional layer to predict answer class, 0: span, 1:yes, 2:no
    with tf.variable_scope("answer_class"):
        # get the representation of CLS
        cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
        cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)
        ans_feature = tf.layers.dense(cls_feature, xlnet_config.d_model,
                                      activation=tf.tanh,
                                      kernel_initializer=initializer,
                                      name='pooler')

        ans_feature = tf.layers.dropout(ans_feature, FLAGS.dropout,
                                        training=is_training)
        cls_logits = tf.layers.dense(ans_feature, 3,  # hotpot has three classes
                                     kernel_initializer=initializer,
                                     name="cls")
        cls_log_probs = tf.nn.log_softmax(cls_logits, -1)
    if is_training:
        return_dict["cls_log_probs"] = cls_log_probs
    else:
        return_dict["cls_logits"] = cls_logits

    return return_dict


def get_model_fn():
    def model_fn(features, labels, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        #### Get loss from inputs
        outputs = get_hotpot_qa_outputs(FLAGS, features, is_training)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        logger.info('#params: {}'.format(num_params))

        scaffold_fn = None

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            if FLAGS.init_checkpoint:
                logger.info(
                    "init_checkpoint not being used in predict mode.")

            predictions = {
                "feature_id": features["feature_id"],
                "start_logits": outputs["start_logits"],
                "end_logits": outputs["end_logits"],
                "cls_logits": outputs["cls_logits"]
            }

            if FLAGS.use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)
            return output_spec

        ### Compute loss
        seq_length = tf.shape(features["input_ids"])[1]

        def compute_loss(log_probs, positions, depth=seq_length):
            one_hot_positions = tf.one_hot(
                positions, depth=depth, dtype=tf.float32)

            loss = - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
            loss = tf.reduce_mean(loss)
            return loss

        start_loss = compute_loss(
            outputs["start_log_probs"], features["answer_start"])
        end_loss = compute_loss(
            outputs["end_log_probs"], features["answer_end"])

        total_loss = (start_loss + end_loss) * 0.5

        cls_loss = compute_loss(
            outputs["cls_log_probs"], features["cls"], depth=3)

        # note(zhiliny): by default multiply the loss by 0.5 so that the scale is
        # comparable to start_loss and end_loss
        total_loss += cls_loss * 0.5

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        #### Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            host_call = function_builder.construct_scalar_host_call(
                monitor_dict=monitor_dict,
                model_dir=FLAGS.model_dir,
                prefix="train/",
                reduce_fn=tf.reduce_mean)

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
    #### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train` and `do_predict` must be True.")
    logger.info("FLAGS: {}".format(FLAGS.flag_values_dict()))

    ### TPU Configuration
    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn()

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if FLAGS.do_train:
        train_input_fn = input_fn_builder(
            input_file=FLAGS.train_record_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
        )

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_predict:
        eval_input_fn = input_fn_builder(
            input_file=FLAGS.eval_record_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False,
        )

        cur_results = []
        for result in estimator.predict(input_fn=eval_input_fn,
                                        yield_single_examples=True):

            if len(cur_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(cur_results)))

            unique_id = int(result["feature_id"])
            start_logits = (
                [float(x) for x in result["start_logits"].flat])
            end_logits = (
                [float(x) for x in result["end_logits"].flat])
            cls_logits = (
                [float(x) for x in result["cls_logits"].flat])
            cur_results.append(
                (unique_id, start_logits, end_logits, cls_logits))

        with tf.gfile.Open(FLAGS.eval_record_file + '.pk', 'wb') as fout:
            pickle.dump(cur_results, fout)
        eval_examples = load_examples(FLAGS.eval_example_file)
        final_cls = collections.OrderedDict()
        final_cls_prob = collections.OrderedDict()
        final_predictions = collections.OrderedDict()
        final_span_scores = collections.OrderedDict()
        all_ground_truths = collections.OrderedDict()
        for cur_result in cur_results:
            unique_id, start_logits, end_logits, cls_logits = cur_result
            item = eval_examples[int(unique_id)]
            orig_id = item['orig_id']

            if 'label' in item:
                answer_cls = item['label']['cls']
                if answer_cls == 0:
                    answers = item['label']['ans']
                    all_ground_truths[orig_id] = [a[1] for a in answers]
                else:
                    answer = 'yes' if answer_cls == 1 else 'no'
                    all_ground_truths[orig_id] = [answer]

            cls_prob = np.exp(cls_logits)
            cls_idx = np.argmax(cls_prob)

            pred_cls_prob = cls_prob[cls_idx]
            if pred_cls_prob > final_cls_prob.get(orig_id, 0):
                final_cls_prob[orig_id] = pred_cls_prob
                final_cls[orig_id] = cls_idx

            if final_cls[orig_id] == 0:
                # or len(item['context_tokens'])
                context_len = len(item['context_spans'])
                context_start = 0  # self.get_context_start(item)
                context_end = context_start + context_len
                # only consider valid context logits
                x_s = np.exp(start_logits[context_start:context_end])
                y_s = np.exp(end_logits[context_start:context_end])
                z = np.outer(x_s, y_s)
                zn = np.tril(np.triu(z), 30)
                pred_start, pred_end = np.unravel_index(np.argmax(zn), zn.shape)
                pred_score = zn[pred_start, pred_end]

                if pred_score > final_span_scores.get(orig_id, 0):
                    start_span = item['context_spans'][pred_start]
                    predicted_char_start = start_span[0]
                    end_span = item['context_spans'][pred_end]
                    predicted_char_end = end_span[1]
                    predicted_text = item['context'][
                                     predicted_char_start:predicted_char_end]
                    final_predictions[orig_id] = predicted_text
                    final_span_scores[orig_id] = pred_score

            elif final_cls[orig_id] == 1:
                # yes
                final_predictions[orig_id] = 'yes'
            else:  # cls == 2
                # no
                final_predictions[orig_id] = 'no'

        pred_path = FLAGS.eval_record_file + '.predictions.json'
        with tf.io.gfile.GFile(pred_path, "w") as f:
            f.write(json.dumps(final_predictions, indent=2) + "\n")
        tf.logging.info("final predictions written to {}".format(pred_path))
        em_score, f1_score = get_em_f1(final_predictions, all_ground_truths)
        tf.logging.info("em={:.4f}, f1={:.4f}".format(em_score, f1_score))


if __name__ == "__main__":
    tf.app.run()
