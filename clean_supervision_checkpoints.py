#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

from merge_checkpoints import gen_var_from_checkpoint
from util import tf


def main(args):
    checkpoint_file = args.checkpoint_file
    out_file = args.out_file
    out_dir = os.path.dirname(out_file)
    if not tf.io.gfile.exists(out_dir):
        tf.io.gfile.makedirs(out_dir)
    with tf.Session() as sess:
        new_vars = []
        print('loading variables from', checkpoint_file)
        for var, var_name in gen_var_from_checkpoint(checkpoint_file):
            # if not re.search(r'(logits/|answer_class/|model/).*', var_name):
            #     continue
            if var_name.startswith('teacher/'):
                continue
            # new_var = tf.Variable(var, name='teacher/{}'.format(var_name))
            new_var = tf.Variable(var, var_name)
            new_vars.append(new_var)

        print('saving weights to:', out_file)
        if not args.dry_run:
            # Save the variables
            saver = tf.train.Saver(new_vars)
            sess.run(tf.global_variables_initializer())
            saver.save(sess, out_file, write_meta_graph=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_file', type=str)
    parser.add_argument('-o', '--out_file', type=str)
    parser.add_argument("-dr", "--dry_run", action='store_true',
                        help="dry run renaming")
    main(parser.parse_args())
