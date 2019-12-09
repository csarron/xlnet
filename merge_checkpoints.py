#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import re

from util import tf


def gen_var_from_checkpoint(checkpoint):
    for var_name, var_shape in tf.train.list_variables(checkpoint):
        var = tf.train.load_variable(checkpoint, var_name)
        if re.search(r'.*Adam.*', var_name):
            continue
        print('{}: {} in {}'.format(var_name, var_shape, checkpoint))
        yield var, var_name


def main(args):
    checkpoint_init = args.checkpoint_init
    checkpoint_teacher = args.checkpoint_teacher
    out_file = args.out_file
    with tf.Session() as sess:
        new_vars = []
        print('loading variables from', checkpoint_init)
        for var, var_name in gen_var_from_checkpoint(checkpoint_init):
            if not var_name.startswith('model/transformer/'):
                continue
            new_var = tf.Variable(var, name=var_name)
            new_vars.append(new_var)

        print('renaming variables from', checkpoint_teacher)
        for var, var_name in gen_var_from_checkpoint(checkpoint_teacher):
            if not re.search(r'(logits/|answer_class/|model/).*', var_name):
                continue
            new_var = tf.Variable(var, name='teacher/{}'.format(var_name))
            new_vars.append(new_var)

        print('saving weights to:', out_file)
        if not args.dry_run:
            # Save the variables
            saver = tf.train.Saver(new_vars)
            sess.run(tf.global_variables_initializer())
            saver.save(sess, out_file, write_meta_graph=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ci', '--checkpoint_init', type=str)
    parser.add_argument('-ct', '--checkpoint_teacher', type=str)
    parser.add_argument('-o', '--out_file', type=str)
    parser.add_argument("-dr", "--dry_run", action='store_true',
                        help="dry run renaming")
    main(parser.parse_args())
