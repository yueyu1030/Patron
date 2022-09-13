task=imdb # dataset
gpu=2
n_gpu=1

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
eval_batch_size=256
dev_labels=32
steps=100
lr=2e-5
batch_size=8
epochs=15

##############################################################################
''' Evalutation on OOD datasets, for IMDB dataset only '''
contra_datasets='sst2val.json,sst2test.json,IMDB-contrast.json,IMDB-counter.json'
extra_cmd="--do_extra_eval --extra_dataset=${contra_datasets}"
##############################################################################

model_type=${model_type} 
output_dir=${task}/model
mkdir -p ${output_dir}
echo ${method}
mkdir -p ${task}/cache
train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py --do_train --do_eval --task=${task} \
	--train_file=train_${train_label}.json --dev_file=valid.json --test_file=test.json \
	--unlabel_file=unlabeled.json \
	--data_dir="${task}" --train_seed=${train_seed} \
	--cache_dir="${task}/cache" \
	--output_dir=${output_dir} --dev_labels=${dev_labels} \
	--gpu=${gpu} --n_gpu=${n_gpu} --num_train_epochs=${epochs} \
	--learning_rate=${lr} --weight_decay=1e-8 \
	--method=${method} --batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} --auto_load=1 \
	--max_steps=${steps} --model_type=${model_type} \
	--sample_labels=${train_label} --al_method=${al_method} ${extra_cmd}"
echo $train_cmd
eval $train_cmd
