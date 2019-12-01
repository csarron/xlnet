python run_squad.py \
  --use_tpu=False \
  --model_config_path="data/squad_ckpt/xlnet_config.json" \
  --spiece_model_file="data/squad_ckpt/spiece.model" \
  --output_dir="data/squad1.1" \
  --init_checkpoint="squad_ckpt/xlnet_model.ckpt" \
  --model_dir="data/squad_ckpt" \
  --predict_file="data/debug-v1.1.json" \
  --eval_record_file="/home/qqcao/work/eet/data/datasets/converted/xlnet/squad_v1.1-dev.10780.tfrecord" \
  --eval_example_file="/home/qqcao/work/eet/data/datasets/converted/xlnet/squad_v1.1-dev.10780.examples.jsonl" \
  --uncased=False \
  --max_seq_length=320 \
  --do_predict=True \
  --predict_batch_size=32 \
  2>&1 | tee data/dev-xlnet-large-squad-v1.1-new.log

python tools/convert_squad.py data/datasets/squad_v1.1/train-v1.1.json data/datasets/eet/squad_v1.1-train.jsonl
python tools/convert_squad.py data/datasets/squad_v1.1/dev-v1.1.json data/datasets/eet/squad_v1.1-dev.jsonl

python tools/convert_squad.py data/datasets/squad_v2.0/train-v2.0.json data/datasets/eet/squad_v2.0-train.jsonl
python tools/convert_squad.py data/datasets/squad_v2.0/dev-v2.0.json data/datasets/eet/squad_v2.0-dev.jsonl


python prepare.py -m bert -s dev -t squad_v2.0 2>&1 | tee data/squad-dev-2.0-bert.log
python prepare.py -m bert -s train -t squad_v2.0 -b 2>&1 | tee data/squad-train-2.0-bert.log
python prepare.py -m bert -s dev -t squad_v1.1 2>&1 | tee data/squad-dev-1.1-bert.log
python prepare.py -m bert -s train -t squad_v1.1 -b 2>&1 | tee data/squad-train-1.1-bert.log

python prepare.py -m xlnet -vf data/res/xlnet_large_cased.spiece.model -s dev -t squad_v2.0 2>&1 | tee data/squad-dev-2.0-xlnet.log
python prepare.py -m xlnet -vf data/res/xlnet_large_cased.spiece.model -s train -t squad_v2.0 2>&1 | tee data/squad-train-2.0-xlnet.log

python prepare.py -m xlnet -vf data/res/xlnet_large_cased.spiece.model -s dev -t squad_v1.1 2>&1 | tee data/squad-dev-1.1-xlnet.log
python prepare.py -m xlnet -vf data/res/xlnet_large_cased.spiece.model -s train -t squad_v1.1 2>&1 | tee data/squad-train-1.1-xlnet.log
