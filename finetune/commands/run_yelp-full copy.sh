task=yelp-full
gpu=0
n_gpu=2

for train_label in 768 1024 1280 1536 1792 2048 4096  ; do
for seed in 0; do
for train_seed in 42 84 128 300  ; do
for label_per_class in 0 ; do
for model_type in roberta-base ; do
for al_method_first in random-1 random-2 random-3  ; do #kmeans_refine1_r0.05 kmeans_refine1_r0.1 kmeans_refine1_r1.0 coreset kmeans kmeansw entropy cal bertkm  bald  kmeans_refine1
for a in 0.5 ; do 
for b in 0.1  ; do 
for c in 0.2 ; do 

al_method=${al_method_first}
# al_method=${al_method_first}_kmeans-dist_${a}_${b}_${c}

seed=${seed}
train_seed=${train_seed}
method=train
max_seq_len=256
self_training_batch_size=16
eval_batch_size=256
dev_labels=100
steps=1000
self_training_max_step=300
logging_steps=20
st_logging_steps=40
epochs=20
if  [ $train_label == 32 ] || [ $train_label == 64 ] || [ $train_label == 128 ]; then
	lr=2e-5
	batch_size=8 
	epochs=15
elif [ $train_label -ge 256 ]; then
	lr=2e-5
	batch_size=32
	epochs=8
else 
	lr=2e-5
	batch_size=4
fi
model_type=${model_type} #dmis-lab/biobert-v1.1 #"allenai/scibert_scivocab_uncased"
# lr_self_training=1e-6
output_dir=../datasets/${task}-0-${seed}/model
mkdir -p ${output_dir}
mkdir -p ../datasets/${task}-${label_per_class}-0/cache
echo ${method}
train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main_unsupervised.py --do_train --do_eval --task=${task} \
	--train_file=train_64.json --dev_file=valid_8.json --test_file=test.json \
	--unlabel_file=unlabeled.json \
	--data_dir="../datasets/${task}-${label_per_class}-${seed}" --seed=${seed} --train_seed=${train_seed} \
	--cache_dir="../datasets/${task}-${label_per_class}-0/cache" \
	--output_dir=${output_dir} \
	--logging_steps=${logging_steps} --self_train_logging_steps=${st_logging_steps} --dev_labels=${dev_labels} \
	--gpu=${gpu} --n_gpu=${n_gpu} --num_train_epochs=${epochs} --weight_decay=1e-8 \
	--learning_rate=${lr} \
	--method=${method} --batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
	--self_training_batch_size=${self_training_batch_size} \
	--max_seq_len=${max_seq_len} --auto_load=1 \
	--self_training_eps=0.8 --max_steps=${steps} --model_type=${model_type} \
	--self_training_update_period=40 --self_training_max_step=${self_training_max_step} \
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