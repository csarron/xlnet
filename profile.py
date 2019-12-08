#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import time

import numpy as np
import tensorflow as tf
from absl import flags
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops

from function_builder import get_qa_outputs
from modeling_decomposed import get_decomposed_qa_outputs
from util import abbreviate
from util import logger


# TODO: remove this fn after TensorFlow merged https://github.com/tensorflow/tensorflow/pull/30575
@ops.RegisterStatistics("BatchMatMulV2", "flops")
def _calc_mat_mul_flops(graph, node):
    """Calculates the compute resources needed for MatMul."""
    transpose_a = node.attr["transpose_a"].b
    a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    a_shape.assert_is_fully_defined()
    if transpose_a:
        k = int(a_shape[-2])
    else:
        k = int(a_shape[-1])
    output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    output_shape.assert_is_fully_defined()
    output_count = np.prod(output_shape.as_list())
    return ops.OpStats("flops", (k * output_count * 2))


flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_integer("max_seq_length",
                     default=320, help="Max sequence length")
flags.DEFINE_integer("max_first_length", default=25,
                     help="max_first_length")
flags.DEFINE_integer("batch_size", default=32,
                     help="batch size for inference")
flags.DEFINE_bool("decompose", default=False, help="whether to use decompose")
flags.DEFINE_integer("sep_layer", default=9,
                     help="which layer to start decompose")
flags.DEFINE_bool("print_parameters", default=False,
                  help="print parameters or not")
flags.DEFINE_bool("profile_memory", default=False,
                  help="whether to profile_memory")
flags.DEFINE_bool("profile_time", default=False,
                  help="whether to profile_time")
flags.DEFINE_bool("not_profile_flops", default=False,
                  help="whether to profile_flops")
flags.DEFINE_integer("cache_segment", default=0,
                     lower_bound=0, upper_bound=2,
                     help="cache first or second segment lower encoding")

FLAGS = flags.FLAGS


def main(_):

    model_name = 'xlnet'
    kwargs = dict(training=False, logits=True)
    batch_size = FLAGS.batch_size
    max_seq_length = FLAGS.max_seq_length

    run_metadata = tf.RunMetadata()
    with tf.Session() as sess:
        if FLAGS.decompose:
            logger.info('running in decompose mode')
            max_first_length = FLAGS.max_first_length
            kwargs['fake_cache_first'] = FLAGS.cache_segment == 1
            kwargs['fake_cache_second'] = FLAGS.cache_segment == 2
            outputs = get_decomposed_qa_outputs(FLAGS, features, False)
        else:
            logger.info('running in normal mode')
            outputs = get_qa_outputs(FLAGS, features, False)

        inputs_dict, logits_ph = model.core_graph(config, **kwargs)

        sess.run(tf.global_variables_initializer())
        opt_builder = tf.profiler.ProfileOptionBuilder
        # saver = tf.train.Saver()
        # saver.save(sess, 'data/sbert', write_meta_graph=False)
        if FLAGS.print_parameters:
            tf.profiler.profile(
                sess.graph,
                options=opt_builder.trainable_variables_parameter())

        if not FLAGS.not_profile_flops:
            prof_options = opt_builder.float_operation()
            prof_options['hide_name_regexes'] = ['.*/Initializer/.*']
            tfprof_node = tf.profiler.profile(sess.graph, options=prof_options)
            profile_metric(model_name, tfprof_node, metric='total_float_ops',
                           metric_name='flops')

        if FLAGS.profile_memory:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = run_metadata
        else:
            options = None
            run_metadata = None
        _ = sess.run([logits_ph], feed_dict=inputs_dict,
                     options=options,
                     run_metadata=run_metadata)

        if FLAGS.profile_memory:
            opts = opt_builder(
                opt_builder.time_and_memory()).build()

            tfprof_node = tf.profiler.profile(
                tf.get_default_graph(),
                run_meta=run_metadata,
                cmd='scope',
                options=opts)

            profile_metric(model_name, tfprof_node,
                           metric='total_requested_bytes', metric_name='mem')

        if FLAGS.profile_time:
            # warm up two rounds
            logger.info("warm up for two rounds...")

            for _ in range(2):
                sess.run([logits_ph], feed_dict=inputs_dict, )

            logger.info("start running 10 rounds...")
            start_time = time.time()
            # bench 10 rounds, take avg
            for _ in range(10):
                sess.run([logits_ph], feed_dict=inputs_dict, )
            end_time = time.time()
            print('infer_time: {:.4f} s'.format((end_time - start_time) / 10))


def profile_metric(model_name, tfprof_node, metric='total_float_ops',
                   metric_name='flops'):
    metric_value = dict()
    # attn_dict = defaultdict(dict)
    attn_other = dict()
    attn_mm = dict()
    attn_attn = dict()
    attn_ctx = dict()
    attn_softmax = dict()
    ffn_metric_other = dict()
    ffn_metric_mm = dict()
    other_metric = dict()

    def traverse_node(node):
        if node.children:
            for child in node.children:
                traverse_node(child)
        else:
            ki = node.name
            vi = getattr(node, metric)
            metric_value[ki] = vi

            if model_name == 'bert':
                attn_pattern = r".*/bert/encoder/layer_\d+/attention.*"
                ffn_pattern = r".*/bert/encoder/layer_\d+/(output|intermediate)/dense.*"
            elif model_name in ['ebert', 'sbert']:
                attn_pattern = r".*/ebert/(upper|lower)_encoder(_1)?/layer_\d+/attention.*"
                ffn_pattern = r".*/ebert/(upper|lower)_encoder(_1)?/layer_\d+/(output|intermediate)/dense.*"
            else:
                # for others
                attn_pattern = ''
                ffn_pattern = ''

            if re.match(attn_pattern, ki):
                if ki.endswith('MatMul'):
                    attn_mm[ki] = vi
                elif ki.endswith('AttnMatmul'):
                    attn_attn[ki] = vi
                elif ki.endswith('ContextMatmul'):
                    attn_ctx[ki] = vi
                elif ki.endswith('Softmax'):
                    attn_softmax[ki] = vi
                else:
                    attn_other[ki] = vi
            elif re.match(ffn_pattern, ki):
                if ki.endswith('MatMul'):
                    ffn_metric_mm[ki] = vi
                else:
                    ffn_metric_other[ki] = vi
            else:
                other_metric[ki] = vi

    traverse_node(tfprof_node)
    total_metric_value = getattr(tfprof_node, metric)
    print()
    print('{}: {}, {}'.format(metric, total_metric_value,
                              abbreviate(total_metric_value)))
    # print('total_sum:', sum(flops.values()))
    attn_mm_sum = sum(attn_mm.values())
    attn_attn_sum = sum(attn_attn.values())
    attn_ctx_sum = sum(attn_ctx.values())
    attn_softmax_sum = sum(attn_softmax.values())
    attn_other_sum = sum(attn_other.values())
    attn_sum = attn_mm_sum + attn_attn_sum + attn_ctx_sum + attn_softmax_sum + attn_other_sum
    print('attn_{}: {}, {}, ({:.2f}%)'.format(metric_name, attn_sum,
                                              abbreviate(attn_sum),
                                              attn_sum * 100 / total_metric_value))
    print('  {}_attn_tran: {}, {}, ({:.2f}%)'.format(metric_name, attn_mm_sum,
                                                     abbreviate(attn_mm_sum),
                                                     attn_mm_sum * 100 / total_metric_value))
    print('  {}_attn_matmul_attn: {}, {}, ({:.2f}%)'.format(metric_name,
                                                            attn_attn_sum,
                                                            abbreviate(
                                                                attn_attn_sum),
                                                            attn_attn_sum * 100 / total_metric_value))
    print('  {}_attn_matmul_ctx: {}, {}, ({:.2f}%)'.format(metric_name,
                                                           attn_ctx_sum,
                                                           abbreviate(
                                                               attn_ctx_sum),
                                                           attn_ctx_sum * 100 / total_metric_value))
    print('  {}_attn_softmax: {}, {}, ({:.2f}%)'.format(metric_name,
                                                        attn_softmax_sum,
                                                        abbreviate(
                                                            attn_softmax_sum),
                                                        attn_softmax_sum * 100 / total_metric_value))
    print(
        '  {}_attn_other: {}, {}, ({:.2f}%)'.format(metric_name, attn_other_sum,
                                                    abbreviate(attn_other_sum),
                                                    attn_other_sum * 100 / total_metric_value))
    ffn_metric_mm_sum = sum(ffn_metric_mm.values())
    ffn_metric_other_sum = sum(ffn_metric_other.values())
    ffn_metric_sum = ffn_metric_mm_sum + ffn_metric_other_sum
    print('ffn_{}: {}, {}, ({:.2f}%)'.format(metric_name, ffn_metric_sum,
                                             abbreviate(ffn_metric_sum),
                                             ffn_metric_sum * 100 / total_metric_value))
    print('  {}_ffn_matmul: {}, {}, ({:.2f}%)'.format(metric_name,
                                                      ffn_metric_mm_sum,
                                                      abbreviate(
                                                          ffn_metric_mm_sum),
                                                      ffn_metric_mm_sum * 100 / total_metric_value))
    print('  {}_ffn_other: {}, {}, ({:.2f}%)'.format(metric_name,
                                                     ffn_metric_other_sum,
                                                     abbreviate(
                                                         ffn_metric_other_sum),
                                                     ffn_metric_other_sum * 100 / total_metric_value))
    print('other_{}: {}, {}, ({:.2f}%)'.format(metric_name,
                                               sum(other_metric.values()),
                                               abbreviate(
                                                   sum(other_metric.values())),
                                               sum(
                                                   other_metric.values()) * 100 / total_metric_value))


if __name__ == "__main__":
    tf.app.run()
