python run_extractive_qa.py \
  --use_tpu=False \
  --model_config_path="data/squad_ckpt/xlnet_config.json" \
  --init_checkpoint="squad_ckpt/xlnet_model.ckpt" \
  --model_dir="data/squad_ckpt" \
  --eval_file="/home/qqcao/work/eet/data/datasets/converted/xlnet/squad_v1.1-dev.10781.tfrecord" \
  --eval_example_file="/home/qqcao/work/eet/data/datasets/converted/xlnet/squad_v1.1-dev.10781.examples.jsonl" \
  --max_seq_length=320 \
  --do_predict=True \
  --predict_batch_size=32 \
  2>&1 | tee data/dev-xlnet-large-squad-v1.1-new2.log

# TODO: eval ckpt at different steps, may tune warmup, train at different msl

ctpu up --tpu-size=v3-8 --tpu-only --name=xlnet-tpu --tf-version 1.14.1.dev20190518 --noconf
sleep 30

msl=1600
model=base
if python run_extractive_qa.py \
  --use_tpu=True --tpu=xlnet-tpu --num_hosts=1 --num_core_per_host=8 \
  --model_config_path="gs://bert-gcs/xlnet/init_cased_${model}/xlnet_config.json" \
  --init_checkpoint="gs://bert-gcs/xlnet/init_cased_${model}/xlnet_model.ckpt" \
  --model_dir="gs://bert-gcs/xlnet/ckpt/hotpot_${model}_${msl}" \
  --train_file="${dataset_dir}/${msl}/hotpot-train.117638.tfrecord" \
  --eval_file="${dataset_dir}/${msl}/hotpot-dev.9687.tfrecord" \
  --eval_example_file="${dataset_dir}/${msl}/hotpot-dev.9687.examples.jsonl" \
  --do_train=True --train_batch_size=16 \
  --do_predict=True --predict_batch_size=16 \
  --learning_rate=3e-5 --adam_epsilon=1e-6 \
  --iterations=1000 --save_steps=1000 \
  --train_steps=15000 --warmup_steps=1000 \
  --max_seq_length=${msl}  2>&1 | tee data/train-xlnet-${model}-hotpot-${msl}.log;test "${PIPESTATUS[0]}"; then

  echo "sucess run, pause tpu"
  ctpu pause  --tpu-only --name=xlnet-tpu --noconf;

else
  echo "run error, not pause tpu"
fi

sleep 120
ctpu pause  --tpu-only --name=xlnet-tpu --noconf;

ctpu up --tpu-size=v3-8 --tpu-only --name=xlnet-tpu --tf-version 1.14.1.dev20190518 --noconf
sleep 30

msl=1600
model=base
dataset_dir="gs://bert-gcs/eet/datasets/converted/xlnet"
for step in 11000 12000 13000 14000; do

  if python run_extractive_qa.py \
    --use_tpu=True --tpu=xlnet-tpu --num_hosts=1 --num_core_per_host=8 \
    --model_config_path="gs://bert-gcs/xlnet/init_cased_${model}/xlnet_config.json" \
    --model_dir="gs://bert-gcs/xlnet/ckpt/hotpot_${model}_${msl}" \
    --checkpoint_path="gs://bert-gcs/xlnet/ckpt/hotpot_${model}_${msl}/model.ckpt-${step}" \
    --eval_file="${dataset_dir}/${msl}/hotpot-dev.9687.tfrecord" \
    --eval_example_file="${dataset_dir}/${msl}/hotpot-dev.9687.examples.jsonl" \
    --max_seq_length=${msl} \
    --do_predict=True --predict_batch_size=16 \
    --iterations=1000 2>&1 | tee data/eval-xlnet-${model}-hotpot-${msl}-ckpt-${step}.log;test "${PIPESTATUS[0]}"; then

    echo "sucess run, pause tpu"

  else
    echo "run error, not pause tpu"
  fi
done

sleep 120
ctpu pause  --tpu-only --name=xlnet-tpu --noconf;


model=xlnet
task=hotpot
for msl in 512 800; do
  python prepare.py -m ${model} -t ${task} -s dev -msl ${msl} 2>&1 | tee data/prep-${task}-${model}-dev-${msl}.log
  python prepare.py -m ${model} -t ${task} -s dev -b -msl ${msl} 2>&1 | tee data/prep-${task}-${model}-devb-${msl}.log
  python prepare.py -m ${model} -t ${task} -s train -b -msl ${msl} 2>&1 | tee data/prep-${task}-${model}-train-${msl}.log
done

gsutil -m cp -r "data/datasets/converted/xlnet/squad_v2.0-train.133637.tfrecord" "gs://bert-gcs/eet/datasets/converted/xlnet/"
gsutil -m cp -r "data/datasets/converted/xlnet/squad_v1.1-dev.10781.tfrecord" "gs://bert-gcs/eet/datasets/converted/xlnet/"
gsutil -m cp -r "data/datasets/converted/xlnet/squad_v1.1-dev.10781.examples.jsonl" "gs://bert-gcs/eet/datasets/converted/xlnet/"

# squad large
msl=320
model=large
dataset_dir="gs://bert-gcs/eet/datasets/converted/xlnet"
if python run_extractive_qa.py --num_classes=2 \
  --use_tpu=True --tpu=xlnet-tpu --num_hosts=1 --num_core_per_host=8 \
  --model_config_path="gs://bert-gcs/xlnet/init_cased_${model}/xlnet_config.json" \
  --init_checkpoint="gs://bert-gcs/xlnet/init_cased_${model}/xlnet_model.ckpt" \
  --model_dir="gs://bert-gcs/xlnet/ckpt/squad_${model}" \
  --train_file="${dataset_dir}/squad_v2.0-train.133637.tfrecord" \
  --eval_file="${dataset_dir}/squad_v1.1-dev.10781.tfrecord" \
  --eval_example_file="${dataset_dir}/squad_v1.1-dev.10781.examples.jsonl" \
  --do_train=True --train_batch_size=32 \
  --do_predict=True --predict_batch_size=32 \
  --learning_rate=3e-5 --adam_epsilon=1e-6 \
  --iterations=1000 --save_steps=1000 \
  --train_steps=15000 --warmup_steps=1000 \
  --max_seq_length=${msl} 2>&1 | tee data/train-xlnet-${model}-squad-${msl}.log;test "${PIPESTATUS[0]}"; then

  echo "sucess run, pause tpu"
  ctpu pause  --tpu-only --name=xlnet-tpu --noconf;

else
  echo "run error, not pause tpu"
fi

sleep 120
ctpu pause  --tpu-only --name=xlnet-tpu --noconf;

model=base
for step in 9000 8000 7000; do
  python run_extractive_qa.py --num_classes=2 --task=squad_v1.1 \
      --use_tpu=True --tpu=xlnet-tpu --num_hosts=1 --num_core_per_host=8 \
      --model_config_path="gs://bert-gcs/xlnet/init_cased_${model}/xlnet_config.json" \
      --model_dir="gs://bert-gcs/xlnet/ckpt/squad_${model}" \
      --checkpoint_path="gs://bert-gcs/xlnet/ckpt/squad_${model}/model.ckpt-${step}" \
      --eval_file="${dataset_dir}/squad_v1.1-dev.10781.tfrecord" \
      --eval_example_file="${dataset_dir}/squad_v1.1-dev.10781.examples.jsonl" \
      --max_seq_length=${msl} --iterations=1000 \
      --do_predict=True --predict_batch_size=16 \
      2>&1 | tee data/eval-xlnet-${model}-hotpot-${msl}-ckpt-${step}.log
done
