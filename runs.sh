module load nvidia 0.495
module load conda 

conda activate 
pip install -r requirements.txt


python -u -m torch.distributed.launch --nproc_per_node 4 transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path ~/deep_learning/BERT-Tickets-DL/full_prune/bert-upper-0.468-lower-0.482 \
  --task_name mnli \
  --tokenizer_name "bert-base-uncased" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir mixed-results/bert-upper-0.468-lower-0.482/mnli

python -u -m torch.distributed.launch --nproc_per_node 4 transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path ~/deep_learning/BERT-Tickets-DL/full_prune/bert-upper-0.468-lower-0.482 \
  --task_name sst2 \
  --tokenizer_name "bert-base-uncased" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --output_dir mixed-results/bert-upper-0.468-lower-0.482/sst2

python -u -m torch.distributed.launch --nproc_per_node 4 transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path ~/deep_learning/BERT-Tickets-DL/full_prune/bert-upper-0.468-lower-0.482 \
  --task_name qnli \
  --tokenizer_name "bert-base-uncased" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir mixed-results/bert-upper-0.468-lower-0.482/qnli

python -u -m torch.distributed.launch --nproc_per_node 4 transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path ~/deep_learning/BERT-Tickets-DL/full_prune/bert-upper-0.468-lower-0.482 \
  --task_name stsb \
  --tokenizer_name "bert-base-uncased" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir mixed-results/bert-upper-0.468-lower-0.482/stsb

python -u -m torch.distributed.launch --nproc_per_node 4 transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path ~/deep_learning/BERT-Tickets-DL/full_prune/bert-upper-0.468-lower-0.482 \
  --task_name mrpc \
  --tokenizer_name "bert-base-uncased" \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir mixed-results/bert-upper-0.468-lower-0.482/mrpc

python -u -m torch.distributed.launch --nproc_per_node 4 transformers/examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path ~/deep_learning/BERT-Tickets-DL/full_prune/bert-upper-0.468-lower-0.482 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --tokenizer_name "bert-base-uncased" \
    --output_dir mixed-results/bert-upper-0.468-lower-0.482/mlm