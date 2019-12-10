"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import modeling
import xlnet
from model_utils import get_train_op
from model_utils import my_init_from_checkpoint
from modeling_decomposed import get_decomposed_qa_outputs
from util import logger
from util import tf


def construct_scalar_host_call(
    monitor_dict,
    model_dir,
    prefix="",
    reduce_fn=None):
    """
    Construct host calls to monitor training progress on TPUs.
    """

    metric_names = list(monitor_dict.keys())

    def host_call_fn(global_step, *args):
        """actual host call function."""
        step = global_step[0]
        with tf.contrib.summary.create_file_writer(
            logdir=model_dir, filename_suffix=".host_call").as_default():
            with tf.contrib.summary.always_record_summaries():
                for i, name in enumerate(metric_names):
                    if reduce_fn is None:
                        scalar = args[i][0]
                    else:
                        scalar = reduce_fn(args[i])
                    with tf.contrib.summary.record_summaries_every_n_global_steps(
                        100, global_step=step):
                        tf.contrib.summary.scalar(prefix + name, scalar,
                                                  step=step)

                return tf.contrib.summary.all_summary_ops()

    global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
    other_tensors = [tf.reshape(monitor_dict[key], [1]) for key in metric_names]

    return host_call_fn, [global_step_tensor] + other_tensors


# def two_stream_loss(FLAGS, features, labels, mems, is_training):
#     """Pretraining loss with two-stream attention Transformer-XL."""
#
#     # ### Unpack input
#     mem_name = "mems"
#     mems = mems.get(mem_name, None)
#
#     inp_k = tf.transpose(features["input_k"], [1, 0])
#     inp_q = tf.transpose(features["input_q"], [1, 0])
#
#     seg_id = tf.transpose(features["seg_id"], [1, 0])
#
#     inp_mask = None
#     perm_mask = tf.transpose(features["perm_mask"], [1, 2, 0])
#
#     if FLAGS.num_predict is not None:
#         # [num_predict x tgt_len x bsz]
#         target_mapping = tf.transpose(features["target_mapping"], [1, 2, 0])
#     else:
#         target_mapping = None
#
#     # target for LM loss
#     tgt = tf.transpose(features["target"], [1, 0])
#
#     # target mask for LM loss
#     tgt_mask = tf.transpose(features["target_mask"], [1, 0])
#
#     # construct xlnet config and save to model_dir
#     xlnet_config = xlnet.XLNetConfig(FLAGS=FLAGS)
#     xlnet_config.to_json(os.path.join(FLAGS.model_dir, "config.json"))
#
#     # construct run config from FLAGS
#     run_config = xlnet.create_run_config(is_training, False, FLAGS)
#
#     xlnet_model = xlnet.XLNetModel(
#         xlnet_config=xlnet_config,
#         run_config=run_config,
#         input_ids=inp_k,
#         seg_ids=seg_id,
#         input_mask=inp_mask,
#         mems=mems,
#         perm_mask=perm_mask,
#         target_mapping=target_mapping,
#         inp_q=inp_q)
#
#     output = xlnet_model.get_sequence_output()
#     new_mems = {mem_name: xlnet_model.get_new_memory()}
#     lookup_table = xlnet_model.get_embedding_table()
#
#     initializer = xlnet_model.get_initializer()
#
#     with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
#         # LM loss
#         lm_loss = modeling.lm_loss(
#             hidden=output,
#             target=tgt,
#             n_token=xlnet_config.n_token,
#             d_model=xlnet_config.d_model,
#             initializer=initializer,
#             lookup_table=lookup_table,
#             tie_weight=True,
#             bi_data=run_config.bi_data,
#             use_tpu=run_config.use_tpu)
#
#     # ### Quantity to monitor
#     monitor_dict = {}
#
#     if FLAGS.use_bfloat16:
#         tgt_mask = tf.cast(tgt_mask, tf.float32)
#         lm_loss = tf.cast(lm_loss, tf.float32)
#
#     total_loss = tf.reduce_sum(lm_loss * tgt_mask) / tf.reduce_sum(tgt_mask)
#     monitor_dict["total_loss"] = total_loss
#
#     return total_loss, new_mems, monitor_dict


# def get_loss(FLAGS, features, labels, mems, is_training):
#     """Pretraining loss with two-stream attention Transformer-XL."""
#     if FLAGS.use_bfloat16:
#         with tf.tpu.bfloat16_scope():
#             return two_stream_loss(FLAGS, features, labels, mems, is_training)
#     else:
#         return two_stream_loss(FLAGS, features, labels, mems, is_training)


def get_classification_loss(FLAGS, features, is_training):
    """Loss for downstream classification tasks."""
    input_ids = features["input_ids"]
    seg_id = features["segment_ids"]
    input_mask_int = tf.cast(tf.cast(input_ids, tf.bool), tf.int32)
    cls_index = tf.reshape(tf.reduce_sum(input_mask_int, axis=1), [-1])
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
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("answer_class"):
            # get the representation of CLS
            cls_index = tf.one_hot(cls_index, seq_len, axis=-1,
                                   dtype=tf.float32)
            cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)
            ans_feature = tf.layers.dense(cls_feature, xlnet_config.d_model,
                                          activation=tf.tanh,
                                          kernel_initializer=initializer,
                                          name='pooler')

            ans_feature = tf.layers.dropout(ans_feature, FLAGS.dropout,
                                            training=is_training)
            # race has 4 classes,
            # boolq has 2 classes
            cls_logits = tf.layers.dense(ans_feature, FLAGS.num_classes,
                                         kernel_initializer=initializer,
                                         name="cls")
            cls_log_probs = tf.nn.log_softmax(cls_logits, -1)
    if is_training:
        return_dict["cls_log_probs"] = cls_log_probs
    return_dict["cls_logits"] = cls_logits

    return return_dict


# def get_regression_loss(
#     FLAGS, features, is_training):
#     """Loss for downstream regression tasks."""
#
#     bsz_per_core = tf.shape(features["input_ids"])[0]
#
#     inp = tf.transpose(features["input_ids"], [1, 0])
#     seg_id = tf.transpose(features["segment_ids"], [1, 0])
#     inp_mask = tf.transpose(features["input_mask"], [1, 0])
#     label = tf.reshape(features["label_ids"], [bsz_per_core])
#
#     xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
#     run_config = xlnet.create_run_config(is_training, True, FLAGS)
#
#     xlnet_model = xlnet.XLNetModel(
#         xlnet_config=xlnet_config,
#         run_config=run_config,
#         input_ids=inp,
#         seg_ids=seg_id,
#         input_mask=inp_mask)
#
#     summary = xlnet_model.get_pooled_out(FLAGS.summary_type,
#                                          FLAGS.use_summ_proj)
#
#     with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
#         per_example_loss, logits = modeling.regression_loss(
#             hidden=summary,
#             labels=label,
#             initializer=xlnet_model.get_initializer(),
#             scope="regression_{}".format(FLAGS.task_name.lower()),
#             return_logits=True)
#
#         total_loss = tf.reduce_mean(per_example_loss)
#
#         return total_loss, per_example_loss, logits


def get_qa_outputs(FLAGS, features, is_training):
    """Loss for downstream span-extraction QA tasks such as SQuAD."""

    input_ids = features["input_ids"]
    seg_id = features["segment_ids"]
    input_mask_int = tf.cast(tf.cast(input_ids, tf.bool), tf.int32)
    cls_index = tf.reshape(tf.reduce_sum(input_mask_int, axis=1), [-1])
    p_mask = tf.cast(tf.cast(seg_id, tf.bool), tf.float32)
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
        # hotpot has 3 classes,
        # squad 2.0 has 2 classes
        cls_logits = tf.layers.dense(ans_feature, FLAGS.num_classes,
                                     kernel_initializer=initializer,
                                     name="cls")
        cls_log_probs = tf.nn.log_softmax(cls_logits, -1)
    if is_training:
        return_dict["cls_log_probs"] = cls_log_probs
    return_dict["cls_logits"] = cls_logits

    return return_dict


def get_race_loss(FLAGS, features, is_training):
    """Loss for downstream multi-choice QA tasks such as RACE."""

    bsz_per_core = tf.shape(features["input_ids"])[0]

    def _transform_features(feature):
        out = tf.reshape(feature, [bsz_per_core, 4, -1])
        out = tf.transpose(out, [2, 0, 1])
        out = tf.reshape(out, [-1, bsz_per_core * 4])
        return out

    inp = _transform_features(features["input_ids"])
    seg_id = _transform_features(features["segment_ids"])
    inp_mask = _transform_features(features["input_mask"])
    label = tf.reshape(features["label_ids"], [bsz_per_core])

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask)
    summary = xlnet_model.get_pooled_out(FLAGS.summary_type,
                                         FLAGS.use_summ_proj)

    with tf.variable_scope("logits"):
        logits = tf.layers.dense(summary, 1,
                                 kernel_initializer=xlnet_model.get_initializer())
        logits = tf.reshape(logits, [bsz_per_core, 4])

        one_hot_target = tf.one_hot(label, 4)
        per_example_loss = -tf.reduce_sum(
            tf.nn.log_softmax(logits) * one_hot_target, -1)
        total_loss = tf.reduce_mean(per_example_loss)

    return total_loss, per_example_loss, logits


def kl_div_loss(student_logits, teacher_logits, temperature=1):
    """The Kullback–Leibler divergence from Q to P:
     D_kl (P||Q) = sum(P * log(P / Q))
    from student to teacher: sum(teacher * log(teacher / student))
    """
    teacher_softmax = tf.nn.softmax(teacher_logits / temperature)
    teacher_log_softmax = tf.nn.log_softmax(teacher_logits / temperature)
    student_log_softmax = tf.nn.log_softmax(student_logits / temperature)
    kl_dist = teacher_softmax * (teacher_log_softmax - student_log_softmax)
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_dist, -1))
    return kl_loss


def get_distill_loss(teacher_outputs, outputs, temp):
    t_start = teacher_outputs['start_logits']
    t_end = teacher_outputs['end_logits']
    t_cls = teacher_outputs['cls_logits']

    s_start = outputs['start_logits']
    s_end = outputs['end_logits']
    s_cls = outputs['cls_logits']
    kd_start_kl_loss = kl_div_loss(s_start, t_start, temp) * (temp ** 2)
    kd_end_kl_loss = kl_div_loss(s_end, t_end, temp) * (temp ** 2)
    kd_cls_kl_loss = kl_div_loss(s_cls, t_cls, temp) * (temp ** 2)
    kd_kl_loss = (kd_start_kl_loss + kd_end_kl_loss + kd_cls_kl_loss) / 3
    return kd_kl_loss


def get_upper_loss(teacher_outputs, outputs):
    t_upper = teacher_outputs['upper_outputs']
    s_upper = outputs['upper_outputs']
    num_upper_layers = len(s_upper)
    t_upper_outputs = t_upper[-num_upper_layers:]

    mse_loss = 0
    for u_output, t_upper_output in zip(s_upper, t_upper_outputs):
        mse_loss += tf.losses.mean_squared_error(u_output,
                                                 t_upper_output)
    mse_loss /= num_upper_layers
    return mse_loss


def get_qa_model_fn(FLAGS):
    def model_fn(features, labels, mode, params):
        # ### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # ### Get loss from inputs
        sep_layer = FLAGS.sep_layer
        if FLAGS.supervise:
            FLAGS.sep_layer = 0  # teacher sep at 0
            logger.info('supervise decompose layer: {}'.format(FLAGS.sep_layer))
            with tf.variable_scope("teacher", reuse=tf.AUTO_REUSE):
                teacher_outputs = get_decomposed_qa_outputs(
                    FLAGS, features, is_training)
        else:
            teacher_outputs = None

        if FLAGS.decompose:
            FLAGS.sep_layer = sep_layer
            logger.info('decompose at layer: {}'.format(FLAGS.sep_layer))
            outputs = get_decomposed_qa_outputs(FLAGS, features, is_training)
        else:
            logger.info('running in normal mode')
            outputs = get_qa_outputs(FLAGS, features, is_training)

        # ### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        logger.info('#params: {}'.format(num_params))

        scaffold_fn = None

        # ### Evaluation mode
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

        # ## Compute loss
        seq_length = FLAGS.max_seq_length

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
            outputs["cls_log_probs"], features["cls"], depth=FLAGS.num_classes)

        # note(zhiliny): by default multiply the loss by 0.5 so that
        # the scale is comparable to start_loss and end_loss
        total_loss += cls_loss * 0.5
        monitor_dict = {"loss/start": start_loss,
                        "loss/end": end_loss,
                        "loss/cls": cls_loss,
                        'loss/ce': total_loss}

        if teacher_outputs is not None:
            ce_loss = total_loss
            gamma = FLAGS.ll_gamma
            alpha = FLAGS.dl_alpha
            beta = FLAGS.ul_beta
            # supervise upper and logits
            temp = FLAGS.temperature
            kd_loss = get_distill_loss(teacher_outputs, outputs, temp)
            mse_loss = get_upper_loss(teacher_outputs, outputs)
            monitor_dict["loss/kd"] = kd_loss
            monitor_dict["loss/mse"] = mse_loss
            total_loss = gamma * ce_loss + alpha * kd_loss + beta * mse_loss
        # ### Configuring the optimizer
        all_trainable_variables = tf.trainable_variables()
        # set fine tune scope
        if FLAGS.tune_scopes:
            tune_scopes = FLAGS.tune_scopes.split(',')
        else:
            tune_scopes = None
        logger.info('tune_scopes: {}'.format(tune_scopes))
        if isinstance(tune_scopes, list):
            scoped_variables = []
            for scope in tune_scopes:
                scoped_variables.extend(tf.trainable_variables(scope))
            trainable_variables = scoped_variables
        else:
            trainable_variables = all_trainable_variables
        if FLAGS.init_scopes:
            init_scopes = FLAGS.init_scopes.split(',')
        else:
            init_scopes = None
        logger.info('init_scopes: {}'.format(init_scopes))
        if isinstance(init_scopes, list):
            to_be_init_variables = []
            for scope in init_scopes:
                to_be_init_variables.extend(tf.trainable_variables(scope))
        else:
            to_be_init_variables = all_trainable_variables

        initialized_variable_names = {}
        scaffold_fn = None
        # ### load pretrained models
        init_checkpoint = FLAGS.init_checkpoint
        if init_checkpoint:
            logger.info("Initialize from the ckpt {}".format(init_checkpoint))
            assign_map, initialized_variable_names = my_init_from_checkpoint(
                init_checkpoint, to_be_init_variables)
            # logger.info('assign_map: \n{}'.format(assign_map))
            # logger.info('initialized_variable_names: \n{}'.format(
            # initialized_variable_names))

            if FLAGS.use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assign_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assign_map)

        logger.info("**** Initialized Variables ****")
        for var in to_be_init_variables:
            init_str = ""
            if var.name in initialized_variable_names:
                init_str = ", *INIT*"
            logger.info("  name=%s, shape=%s%s",
                        var.name, var.shape, init_str)

        if mode == tf.estimator.ModeKeys.TRAIN:
            logger.info("**** Trainable Variables ****")
            for var in trainable_variables:
                init_str = ""
                if var.name in initialized_variable_names:
                    init_str = ", *INIT_AND_TRAINABLE*"
                logger.info("*TRAINABLE*  name=%s, shape=%s%s",
                            var.name, var.shape, init_str)
        train_op, learning_rate, _ = get_train_op(
            FLAGS, total_loss, trainable_variables=trainable_variables)

        monitor_dict["lr"] = learning_rate

        # ### Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            host_call = construct_scalar_host_call(
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


def qa_input_fn_builder(FLAGS, is_training,
                        num_threads=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    input_file = FLAGS.train_file if is_training else FLAGS.eval_file
    if FLAGS.decompose:
        max_q_length = FLAGS.max_first_length + 2
        max_c_length = FLAGS.max_seq_length - max_q_length
        name_to_features = {
            "feature_id": tf.io.FixedLenFeature([], tf.int64),
            "question_ids": tf.io.FixedLenFeature([max_q_length], tf.int64),
            "context_ids": tf.io.FixedLenFeature([max_c_length], tf.int64),
        }
    else:
        seq_length = FLAGS.max_seq_length
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
                drop_remainder=is_training))
        d = d.prefetch(1024)

        return d

    return input_fn
