from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xlnet
from modeling import embedding_lookup
from modeling import positionwise_ffn
from modeling import rel_multihead_attn
from modeling import relative_positional_encoding
from util import logger
from util import tf


def transformer_xl_decomposed(n_token, n_layer, d_model, n_head,
                              d_head, d_inner, dropout, dropatt, attn_type,
                              is_training, initializer, q_ids, ctx_ids,
                              clamp_len=-1, untie_r=False, use_tpu=True,
                              ff_activation='relu', use_bfloat16=False,
                              sep_layer=9, q_attn_mask=None,
                              c_attn_mask=None, qc_attn_mask=None,
                              q_seq_len=None, ctx_seq_len=None,
                              scope='transformer', **kwargs):
    tf_float = tf.bfloat16 if use_bfloat16 else tf.float32
    logger.info('Use float type {}'.format(tf_float))
    # new_mems = []
    with tf.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
        else:
            r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                       dtype=tf_float, initializer=initializer)

        # batch_size = tf.shape(input_ids)[1]
        # seq_len = tf.shape(input_ids)[0]
        batch_size = tf.shape(q_ids)[1]

        # mlen = tf.shape(mems[0])[0] if mems is not None else 0
        # mlen = 0
        # klen = mlen + seq_len

        # #### Attention mask
        attn_mask = None

        # data_mask = input_mask[None]
        # if data_mask is not None:
        # all mems can be attended to
        # mems_mask = tf.zeros([tf.shape(data_mask)[0], mlen, batch_size],
        #                      dtype=tf_float)
        # data_mask = tf.concat([mems_mask, data_mask], 1)
        # if attn_mask is None:
        #     attn_mask = data_mask[:, :, :, None]
        # else:
        #     attn_mask += data_mask[:, :, :, None]
        # non_tgt_mask = None

        # #### Word embedding
        q_emb, lookup_table = embedding_lookup(
            x=q_ids, n_token=n_token, d_embed=d_model,
            initializer=initializer,
            use_tpu=use_tpu, dtype=tf_float, scope='word_embedding')

        c_emb, _ = embedding_lookup(
            x=ctx_ids, n_token=n_token, d_embed=d_model,
            initializer=initializer,
            use_tpu=use_tpu, dtype=tf_float,
            reuse=True, scope='word_embedding')

        q_output_h = tf.layers.dropout(q_emb, dropout, training=is_training)
        ctx_output_h = tf.layers.dropout(c_emb, dropout, training=is_training)

        # #### Segment embedding
        if untie_r:
            r_s_bias = tf.get_variable('r_s_bias',
                                       [n_layer, n_head, d_head],
                                       dtype=tf_float,
                                       initializer=initializer)
        else:
            # default case (tie)
            r_s_bias = tf.get_variable('r_s_bias', [n_head, d_head],
                                       dtype=tf_float,
                                       initializer=initializer)

        seg_embed = tf.get_variable('seg_embed',
                                    [n_layer, 2, n_head, d_head],
                                    dtype=tf_float, initializer=initializer)

        # Convert `seg_id` to one-hot `seg_mat`
        # mem_pad = tf.zeros([mlen, batch_size], dtype=tf.int32)
        # cat_ids = tf.concat([mem_pad, seg_id], 0)

        # `1` indicates not in the same segment [qlen x klen x bsz]
        ctx_seg_ids = tf.zeros_like(ctx_ids, dtype=tf.int32)
        ctx_seg_mat = tf.cast(tf.logical_not(tf.equal(ctx_seg_ids[:, None],
                                                      ctx_seg_ids[None, :])),
                              tf.int32)
        ctx_seg_mat = tf.one_hot(ctx_seg_mat, 2, dtype=tf_float)
        q_seg_ids = tf.ones_like(q_ids, dtype=tf.int32)
        q_seg_mat = tf.cast(tf.logical_not(tf.equal(q_seg_ids[:, None],
                                                    q_seg_ids[None, :])),
                            tf.int32)
        q_seg_mat = tf.one_hot(q_seg_mat, 2, dtype=tf_float)

        seg_ids = tf.concat([ctx_seg_ids, q_seg_ids], axis=0)
        seg_mat = tf.cast(
            tf.logical_not(tf.equal(seg_ids[:, None], seg_ids[None, :])),
            tf.int32)
        seg_mat = tf.one_hot(seg_mat, 2, dtype=tf_float)

        # #### Positional encoding
        # q_pos_emb = relative_positional_encoding(
        #     q_seq_len, q_seq_len, d_model, clamp_len, attn_type,
        #     bsz=batch_size, dtype=tf_float)
        # q_pos_emb = tf.layers.dropout(q_pos_emb, dropout,
        # training=is_training)
        #
        # ctx_pos_emb = relative_positional_encoding(
        #     ctx_seq_len, ctx_seq_len, d_model, clamp_len, attn_type,
        #     bsz=batch_size, dtype=tf_float)
        # ctx_pos_emb = tf.layers.dropout(ctx_pos_emb, dropout,
        #                                 training=is_training)
        # pos_emb = tf.concat([ctx_pos_emb, q_pos_emb], axis=0)
        seq_len = ctx_seq_len + q_seq_len
        pos_emb = relative_positional_encoding(
            seq_len, seq_len, d_model, clamp_len, attn_type,
            bsz=batch_size, dtype=tf_float)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)
        ctx_pos_emb = pos_emb[q_seq_len:q_seq_len + 2 * ctx_seq_len, :, :]
        q_pos_emb1 = pos_emb[:q_seq_len, :, :]
        q_pos_emb2 = pos_emb[q_seq_len + 2 * ctx_seq_len:, :, :]
        q_pos_emb = tf.concat([q_pos_emb1, q_pos_emb2], axis=0)
        # #### Attention layers
        # mems = [None] * n_layer
        for i in range(sep_layer):
            r_s_bias_i = r_s_bias if not untie_r else r_s_bias[i]
            r_w_bias_i = r_w_bias if not untie_r else r_w_bias[i]
            r_r_bias_i = r_r_bias if not untie_r else r_r_bias[i]
            seg_embed_i = seg_embed[i]
            with tf.variable_scope('layer_{}'.format(i)):
                ctx_output_h = rel_multihead_attn(
                    h=ctx_output_h,
                    r=ctx_pos_emb,
                    r_w_bias=r_w_bias_i,
                    r_r_bias=r_r_bias_i,
                    r_s_bias=r_s_bias_i,
                    seg_mat=ctx_seg_mat,
                    seg_embed=seg_embed_i,
                    attn_mask=c_attn_mask,
                    mems=None,
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer,
                    reuse=False)

                ctx_output_h = positionwise_ffn(
                    inp=ctx_output_h,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    activation_type=ff_activation,
                    is_training=is_training,
                    reuse=False)

                q_output_h = rel_multihead_attn(
                    h=q_output_h,
                    r=q_pos_emb,
                    r_w_bias=r_w_bias_i,
                    r_r_bias=r_r_bias_i,
                    r_s_bias=r_s_bias_i,
                    seg_mat=q_seg_mat,
                    seg_embed=seg_embed_i,
                    attn_mask=q_attn_mask,
                    mems=None,
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer,
                    reuse=True)

                q_output_h = positionwise_ffn(
                    inp=q_output_h,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    activation_type=ff_activation,
                    is_training=is_training,
                    reuse=True)

        # concat all q, ctx related variables
        output_h = tf.concat([ctx_output_h, q_output_h], axis=0)
        upper_outputs = []
        for i in range(sep_layer, n_layer):
            r_s_bias_i = r_s_bias if not untie_r else r_s_bias[i]
            r_w_bias_i = r_w_bias if not untie_r else r_w_bias[i]
            r_r_bias_i = r_r_bias if not untie_r else r_r_bias[i]
            seg_embed_i = seg_embed[i]
            with tf.variable_scope('layer_{}'.format(i)):
                output_h = rel_multihead_attn(
                    h=output_h,
                    r=pos_emb,
                    seg_mat=seg_mat,
                    r_w_bias=r_w_bias_i,
                    r_r_bias=r_r_bias_i,
                    r_s_bias=r_s_bias_i,
                    seg_embed=seg_embed_i,
                    attn_mask=qc_attn_mask,
                    mems=None,
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer,
                    reuse=False)

                output_h = positionwise_ffn(
                    inp=output_h,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    activation_type=ff_activation,
                    is_training=is_training,
                    reuse=False)
                upper_outputs.append(output_h)
        output = tf.layers.dropout(output_h, dropout, training=is_training)
        upper_outputs[-1] = output
        return upper_outputs


def get_attention_mask(input_ids, seq_len):
    # `non_tgt_mask` = [Seq_len, Seq_len]
    non_tgt_mask = -tf.eye(seq_len, dtype=tf.float32)
    # `input_mask` = [Seq_len, Batch_size]
    input_mask = 1 - tf.cast(tf.cast(input_ids, tf.bool), tf.float32)
    # `data_mask` = [1, Seq_len, Batch_size]
    data_mask = input_mask[None]
    # `attn_mask` = [1, Seq_len, Batch_size, 1]
    attn_mask = data_mask[:, :, :, None]
    # `non_tgt_mask` = [Seq_len, Seq_len, Batch_size, 1]
    non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                           dtype=tf.float32)
    return non_tgt_mask


def get_decomposed_qa_outputs(FLAGS, features, is_training):
    question_ids = features["question_ids"]
    context_ids = features["context_ids"]
    seq_len = FLAGS.max_seq_length
    q_seq_len = FLAGS.max_first_length + 2
    ctx_seq_len = seq_len - q_seq_len
    q_mask_int = tf.cast(tf.cast(question_ids, tf.bool), tf.int32)
    cls_index = tf.reshape(tf.reduce_sum(q_mask_int, axis=1) + ctx_seq_len,
                           [-1])
    # 0 for mask out
    # q_zeros = tf.zeros_like(question_ids)
    # p_ids = tf.concat([context_ids, q_zeros], axis=1)
    # p_mask = tf.cast(tf.cast(p_ids, tf.bool), tf.float32)
    question_ids = tf.transpose(question_ids, [1, 0])
    context_ids = tf.transpose(context_ids, [1, 0])

    q_attn_mask = get_attention_mask(question_ids, q_seq_len)
    c_attn_mask = get_attention_mask(context_ids, ctx_seq_len)
    qc_attn_mask = get_attention_mask(tf.concat([context_ids, question_ids],
                                                axis=0), seq_len)

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)
    initializer = xlnet._get_initializer(run_config)
    tfm_args = dict(
        n_token=xlnet_config.n_token,
        initializer=initializer,
        attn_type="bi",
        n_layer=xlnet_config.n_layer,
        d_model=xlnet_config.d_model,
        n_head=xlnet_config.n_head,
        d_head=xlnet_config.d_head,
        d_inner=xlnet_config.d_inner,
        ff_activation=xlnet_config.ff_activation,
        untie_r=xlnet_config.untie_r,

        is_training=run_config.is_training,
        use_bfloat16=run_config.use_bfloat16,
        use_tpu=run_config.use_tpu,
        dropout=run_config.dropout,
        dropatt=run_config.dropatt,

        # mem_len=run_config.mem_len,
        # reuse_len=run_config.reuse_len,
        # bi_data=run_config.bi_data,
        clamp_len=run_config.clamp_len,
        # same_length=run_config.same_length,
        ctx_ids=context_ids,
        q_ids=question_ids,
        q_seq_len=q_seq_len,
        ctx_seq_len=ctx_seq_len,
        sep_layer=FLAGS.sep_layer,
        q_attn_mask=q_attn_mask,
        c_attn_mask=c_attn_mask,
        qc_attn_mask=qc_attn_mask,
    )

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        upper_outputs = transformer_xl_decomposed(**tfm_args)

    output = upper_outputs[-1]
    return_dict = {'upper_outputs': upper_outputs}
    with tf.variable_scope("logits"):
        # logits: seq, batch_size, 2
        logits = tf.layers.dense(output, 2, kernel_initializer=initializer)

        # logits: 2, batch_size, seq
        logits = tf.transpose(logits, [2, 1, 0])

        # start_logits: batch_size, seq
        # end_logits: batch_size, seq
        start_logits, end_logits = tf.unstack(logits, axis=0)

        # start_logits_masked = start_logits * p_mask - 1e30 * (1 - p_mask)
        # start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)
        start_log_probs = tf.nn.log_softmax(start_logits, -1)

        # end_logits_masked = end_logits * p_mask - 1e30 * (1 - p_mask)
        # end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
        end_log_probs = tf.nn.log_softmax(end_logits, -1)

    return_dict["start_logits"] = start_logits
    return_dict["end_logits"] = end_logits
    if is_training:
        return_dict["start_log_probs"] = start_log_probs
        return_dict["end_log_probs"] = end_log_probs

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

    return_dict["cls_logits"] = cls_logits
    if is_training:
        return_dict["cls_log_probs"] = cls_log_probs

    return return_dict
