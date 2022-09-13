task=imdb
gpu=0 # the id of the GPU

train_label=32 # number of labels 32/64/128
train_seed=128 # random seed
model_type=roberta-base
rho=0.1
gamma=0.3
beta=1  
mu=0.5

al_method=patron_rho${rho}_gamma${gamma}_beta${beta}_mu${mu}

method=train
max_seq_len=256
eval_batch_size=128
dev_labels=100
steps=100
epochs=15
lr=1e-5
batch_size=16

model_type=${model_type} 
output_dir=${task}/model
mkdir -p ${output_dir}
echo ${method}
mkdir -p ${task}/pt_cache

train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 LMBFF.py --do_train --do_eval --task=${task} \
	--train_file=train_8.json --dev_file=valid.json --test_file=test.json \
	--unlabel_file=unlabeled.json \
	--data_dir=${task} --train_seed=${train_seed} \
	--cache_dir=${task}/pt_cache \
	--output_dir=${output_dir} --lr=${lr} \
	--logging_steps=${logging_steps} --dev_labels=${dev_labels} \
	--num_train_epochs=${epochs} \
	--batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} \
	--max_steps=${steps} --model_type=${model_type} \
	--sample_labels=${train_label} --al_method=${al_method}"
echo $train_cmd
eval $train_cmd
