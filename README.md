# Multilingual-Complex-Named-Entity-Recognition


WIP

```bash
# bert-base-cased
python run_flax_ner.py \
  --model_name_or_path bert-base-cased \
  --train_file training_data_json/EN-English/train.json \
  --validation_file training_data_json/EN-English/dev.json \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --output_dir models/bert-ner-conll2003 \
  --eval_steps 300
```

```bash
# bert-large-cased
python run_flax_ner.py \
  --model_name_or_path bert-large-cased \
  --train_file training_data_json/EN-English/train.json \
  --validation_file training_data_json/EN-English/dev.json \
  --max_seq_length 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --output_dir models/bert-ner-conll2003 \
  --eval_steps 300
```


```bash
python run_flax_predict.py \
--model_name_or_path models/bert-ner-conll2003 \
--test_file training_data_json/EN-English/dev.json \
--max_seq_length 128 \
--output_file en.dev.conll
```