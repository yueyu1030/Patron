task=imdb
gpu=2
n_gpu=2

for train_label in 64; do #  8 16 32 64
for seed in 0  ; do
for train_seed in 200 42 100 128 87 0; do # 128 0 21 42 87 100
for label_per_class in 0 ; do
for model_type in roberta-base ; do
# 64: kmeans_refine1_r0.05 kmeans_refine1_r0.1 
# 128: margin cal margin-km random-1
for al_method_first in  coreset kmeans kmeansw tpc  ; do #kmeans_refine1 entropy cal bertkm coreset bald badge kmeansw tpc 
for a in  0.5; do  #0.3 0.1
for b in 0.5; do #0.1 0.5 0 1 5 0.5_0.5 0.2_0.5 0.01_0.1 0.1_0.1 0_0.1 5_0.1 0.5_0.2 
for c in 0.2 ; do
# for seed in 0  ; do
# for train_seed in 0 42 84 128 ; do
# for label_per_class in 0 ; do
# for model_type in bert-basse-uncased ; do # bert-base-uncased bert-base-uncased roberta-base roberta-large 
# for al_method in random0 random1 random2 random3 ; do #cal bertkm coreset bald badge entropy
al_method=${al_method_first}
# al_method=${al_method_first}_kmeans-dist_${a}_${b}_${c}
seed=${seed}
train_seed=${train_seed}
method=train
max_seq_len=256
self_training_batch_size=16
eval_batch_size=256
dev_labels=100
steps=100
self_training_max_step=300
logging_steps=20
st_logging_steps=40
epochs=18
contra_datasets='sst2val.json,sst2test.json,IMDB-contrast.json,IMDB-counter.json'
extra_cmd="--do_extra_eval --extra_dataset=${contra_datasets}"
if [ $train_label == 32 ] ; then
	lr=1e-5
	batch_size=8
	epochs=15
elif  [ $train_label == 64 ] || [ $train_label == 128 ] ; then 
	lr=2e-5
	batch_size=16
else
	lr=3e-5
	batch_size=8
	epochs=15
fi
model_type=${model_type} #dmis-lab/biobert-v1.1 #"allenai/scibert_scivocab_uncased"
# lr_self_training=1e-6
output_dir=./results/${task}-0-0/model
mkdir -p ${output_dir}
echo ${method}
mkdir -p ../datasets/${task}-${label_per_class}-0/pt_cache
# valid_${train_label}.json
train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 LMBFF.py --do_train --do_eval --task=${task} \
	--train_file=train_8.json --dev_file=valid_16.json --test_file=test.json \
	--unlabel_file=unlabeled.json \
	--data_dir=../datasets/${task}-${label_per_class}-0 --seed=${seed} --train_seed=${train_seed} \
	--cache_dir=../datasets/${task}-${label_per_class}-0/pt_cache \
	--output_dir=${output_dir} --lr=${lr} \
	--logging_steps=${logging_steps} --self_train_logging_steps=${st_logging_steps} --dev_labels=${dev_labels} \
	--num_train_epochs=${epochs} \
	--batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--max_seq_len=${max_seq_len} \
	--max_steps=${steps} --model_type=${model_type} \
	--sample_labels=${train_label} --al_method=${al_method}"
echo $train_cmd
eval $train_cmd
done
done
done
done
done
done

done 
done
done