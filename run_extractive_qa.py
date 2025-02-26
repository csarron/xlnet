# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import numpy as np

import function_builder
import model_utils
from eval import get_em_f1
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

# Data preprocessing config
flags.DEFINE_integer("max_seq_length",
                     default=512, help="Max sequence length")
flags.DEFINE_integer("doc_stride",
                     default=128, help="Doc stride")
flags.DEFINE_integer("max_answer_length",
                     default=30, help="Max answer length")
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
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_batch_size", default=48,
                     help="batch size for training")
flags.DEFINE_integer("train_steps", default=8000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("max_save", default=8,
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
flags.DEFINE_integer("num_classes", default=2,
                     help="batch size for prediction")
flags.DEFINE_string("task", default="squad_v1.1",
                    help="squad_v1.1 or squad_v2.0 or hotpot")
flags.DEFINE_string("prediction_file", default="",
                    help="prediction path prefix")
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


def main(_):
    FLAGS.use_tpu = True if FLAGS.tpu else False
    # ### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train` and `do_predict` must be True.")
    logger.info("FLAGS: {}".format(FLAGS.flag_values_dict()))

    # ## TPU Configuration
    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = function_builder.get_qa_model_fn(FLAGS)

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
        train_input_fn = function_builder.qa_input_fn_builder(
            FLAGS, is_training=True)

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_predict:
        eval_input_fn = function_builder.qa_input_fn_builder(
            FLAGS, is_training=False)

        cur_results = []
        checkpoint_path = FLAGS.checkpoint_path
        if not checkpoint_path:
            checkpoint_path = None
        for result in estimator.predict(input_fn=eval_input_fn,
                                        checkpoint_path=checkpoint_path,
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

        eval_examples = load_examples(FLAGS.eval_example_file)
        final_cls = collections.OrderedDict()
        final_cls_prob = collections.OrderedDict()
        final_predictions = collections.OrderedDict()
        final_span_scores = collections.OrderedDict()
        all_ground_truths = collections.OrderedDict()
        pred_info = collections.defaultdict(dict)
        for cur_result in cur_results:
            unique_id, start_logits, end_logits, cls_logits = cur_result
            item = eval_examples[int(unique_id)]
            orig_id = item['orig_id']
            # save cls scores and spans scores for tune final outputs
            pred_item = {'orig_id': orig_id,
                         'pred_cls_scores': [float(s) for s in
                                             np.exp(cls_logits)]}

            # final_cls[orig_id] == 0
            context_len = len(item['context_spans'])
            context_start = 0  # self.get_context_start(item)
            context_end = context_start + context_len
            # only consider valid context logits
            x_s = np.exp(start_logits[context_start:context_end])
            y_s = np.exp(end_logits[context_start:context_end])
            z = np.outer(x_s, y_s)
            zn = np.tril(np.triu(z), FLAGS.max_answer_length)
            pred_start, pred_end = np.unravel_index(np.argmax(zn), zn.shape)
            pred_score = zn[pred_start, pred_end]
            pred_item['pred_score'] = pred_score
            pred_item['pred_span'] = [pred_start, pred_end]
            if pred_score > final_span_scores.get(orig_id, 0):
                start_span = item['context_spans'][pred_start]
                predicted_char_start = start_span[0]
                end_span = item['context_spans'][pred_end]
                predicted_char_end = end_span[1]
                predicted_text = item['context'][
                                 predicted_char_start:predicted_char_end]
                pred_item['pred_text'] = predicted_text.strip()
                final_predictions[orig_id] = predicted_text.strip()
                final_span_scores[orig_id] = pred_score

            if 'label' in item:
                answer_cls = item['label']['cls']
                pred_item['label_cls'] = answer_cls
                if answer_cls == 0:
                    answers = item['label']['ans']
                    answer_texts = [a[1] for a in answers]
                    all_ground_truths[orig_id] = answer_texts
                    pred_item['label_span'] = answer_texts
                else:
                    answer = 'yes' if answer_cls == 1 else 'no'
                    all_ground_truths[orig_id] = [answer]
                    pred_item['label_span'] = [answer]

            cls_prob = np.exp(cls_logits)
            cls_idx = np.argmax(cls_prob)

            pred_cls_prob = cls_prob[cls_idx]
            if pred_cls_prob > final_cls_prob.get(orig_id, 0):
                final_cls_prob[orig_id] = pred_cls_prob
                final_cls[orig_id] = cls_idx

            if final_cls[orig_id] == 1:
                # yes for hotpot, impossible for squad 2.0
                if FLAGS.task == 'squad_v2.0':
                    final_predictions[orig_id] = ''
                elif FLAGS.task == 'hotpot':
                    final_predictions[orig_id] = 'yes'
            elif final_cls[orig_id] == 2:  # cls == 2
                # no
                final_predictions[orig_id] = 'no'

            pred_info[int(unique_id)] = pred_item

        ckpt = os.path.basename(checkpoint_path) if checkpoint_path else ''
        dec_suffix = '_s{}'.format(FLAGS.sep_layer) if FLAGS.decompose else ''
        prediction_prefix = FLAGS.eval_file + ckpt + dec_suffix + '.predictions'
        prediction_file = prediction_prefix + '.json'
        pred_path = FLAGS.prediction_file or prediction_file
        pred_dir = os.path.dirname(pred_path)
        if not tf.io.gfile.exists(pred_dir):
            tf.io.gfile.makedirs(pred_dir)
        if FLAGS.task == 'hotpot':
            final_predictions_data = {"answer": final_predictions, "sp": {}}
        else:
            final_predictions_data = final_predictions
        with tf.io.gfile.GFile(pred_path, "w") as f:
            f.write(json.dumps(final_predictions_data, indent=2,
                               ensure_ascii=False) + "\n")
        logger.info("final predictions written to {}".format(pred_path))
        em_score, f1_score = get_em_f1(final_predictions, all_ground_truths)
        logger.info("em={:.4f}, f1={:.4f}".format(em_score, f1_score))


if __name__ == "__main__":
    tf.app.run()
