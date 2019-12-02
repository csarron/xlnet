python run_squad.py \
  --use_tpu=False \
  --model_config_path="data/squad_ckpt/xlnet_config.json" \
  --spiece_model_file="data/squad_ckpt/spiece.model" \
  --init_checkpoint="squad_ckpt/xlnet_model.ckpt" \
  --model_dir="data/squad_ckpt" \
  --eval_record_file="/home/qqcao/work/eet/data/datasets/converted/xlnet/squad_v1.1-dev.10781.tfrecord" \
  --eval_example_file="/home/qqcao/work/eet/data/datasets/converted/xlnet/squad_v1.1-dev.10781.examples.jsonl" \
  --uncased=False \
  --max_seq_length=320 \
  --do_predict=True \
  --predict_batch_size=32 \
  2>&1 | tee data/dev-xlnet-large-squad-v1.1-new2.log

ctpu up --tpu-size=v3-8 --tpu-only --name=bert-tpu --tf-version 1.14.1.dev20190518 --noconf

if python run_hotpot.py \
  --use_tpu=True \
  --tpu=bert-tpu \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --model_config_path="gs://bert-gcs/xlnet/init_cased_large/xlnet_config.json" \
  --spiece_model_file=data/spiece.model \
  --init_checkpoint="gs://bert-gcs/xlnet/init_cased_large/xlnet_model.ckpt" \
  --model_dir="gs://bert-gcs/xlnet/data/hotpot_ckpt" \
  --train_file="gs://bert-gcs/eet/datasets/converted/xlnet/hotpot-train.94701.tfrecord" \
  --eval_record_file="gs://bert-gcs/eet/datasets/converted/xlnet/hotpot-dev.7742.tfrecord" \
  --eval_example_file="gs://bert-gcs/eet/datasets/converted/xlnet/hotpot-dev.7742.examples.jsonl" \
  --uncased=False \
  --max_seq_length=2048 \
  --do_train=True \
  --train_batch_size=32 \
  --do_predict=True \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=1000 \
  --train_steps=8000 \
  --warmup_steps=1000 2>&1 | tee data/tune-xlnet-large-hotpot.log;test "${PIPESTATUS[0]}"; then
  echo "sucess run, pause tpu"
  ctpu pause  --tpu-only --name=bert-tpu --noconf;
else
  echo "run error, not pause tpu"
fi

python run_squad.py \
  --use_tpu=True \
  --tpu=bert-tpu \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --model_config_path="gs://bert-gcs/xlnet/init_cased_large/xlnet_config.json" \
  --spiece_model_file=data/spiece.model \
  --output_dir="gs://bert-gcs/xlnet/data/squad1.1" \
  --init_checkpoint="gs://bert-gcs/xlnet/init_cased_large/xlnet_model.ckpt" \
  --model_dir="gs://bert-gcs/xlnet/data/squad_ckpt" \
  --predict_file="gs://bert-gcs/xlnet/data/squad1.1/dev-v1.1.json" \
  --uncased=False \
  --max_seq_length=384 \
  --do_train=False \
  --train_batch_size=48 \
  --do_predict=True \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=1000 \
  --train_steps=8000 \
  --warmup_steps=1000 2>&1 | tee data/eval-xlnet-large-squad-v1.1.log
