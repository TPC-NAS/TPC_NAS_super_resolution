#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

# budget_flops=100e9
budget_model_size=500e3
max_layers=50
input_image_size=480
population_size=128
evolution_max_iter=10000

save_dir=./save_dir/Zen_NAS_params500_final_test
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,8,1,8,1)SuperResK3K3(8,8,1,8,1)SuperResK3K3(8,8,1,8,1)SuperResK3K3(8,8,1,8,1)" \
> ${save_dir}/init_plainnet.txt

export CUDA_VISIBLE_DEVICES=0

python evolution_search.py --gpu 0 \
  --zero_shot_score TPC_hardware --scale 2 --channel 64 \
  --search_space SearchSpace/search_space_XXBL.py \
  --budget_model_size ${budget_model_size} \
  --max_layers ${max_layers} \
  --batch_size 64 \
  --input_image_size ${input_image_size} \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 100 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}

python analyze_model.py \
  --input_image_size ${input_image_size} --scale 2 \
  --num_classes 100 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --save_dir ${save_dir}

python train_super_resolution.py \
  --dist_mode single --workers_per_gpu 6 --no_BN True --scale 2 --use_se --auto_resume \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 64 \
  --save_dir ${save_dir}/cifar100_120epochs

python test_multi_super_resolution.py \
  --dist_mode single --workers_per_gpu 6 --no_BN True --scale 2 --use_se \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 64 \
  --save_dir ${save_dir}/cifar100_120epochs

# python new_train.py \
#   --arch Masternet.py:MasterNet --scale 2 --no_BN True \
#   --plainnet_struct_txt ${save_dir}/best_structure.txt \
#   --save_dir ${save_dir}/cifar100_120epochs --cuda --resume "/home/user1/ZenNAS_super_resolution/save_dir/Zen_NAS_cifar_flops200_hardware/cifar100_120epochs/model_ch[RGB][RGB]_epoch_latest.pth" \
