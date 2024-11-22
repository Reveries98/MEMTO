# # SMAP
# python main.py --anormly_ratio 1.0 \
#                 --num_epochs 200 \
#                 --batch_size 256 \
#                 --mode train \
#                 --dataset SMAP \
#                 --data_path /media/media02/mhzheng/dataset/SMAP \
#                 --input_c 25 \
#                 --output_c 25 \
#                 --n_memory 10 \
#                 --lambd 0.005 \
#                 --lr 5e-5 \
#                 --memory_initial False \
#                 --phase_type None \
#                 --ffn_ratio 1 \
#                 --patch_size 1 \
#                 --patch_stride 1 \
#                 --num_blocks 1 \
#                 --large_size 13 \
#                 --small_size 5 \
#                 --dims 128 \
#                 --head_dropout 0.0 \
#                 --dropout 0.1 \
#                 --pred_len 0\
#                 --backbone ModernTCN

# python main.py --anormly_ratio 1.0 \
#                 --num_epochs 200  \
#                 --batch_size 256  \
#                 --mode memory_initial \
#                 --dataset SMAP  \
#                 --data_path /media/media02/mhzheng/dataset/SMAP  \
#                 --input_c 25 \
#                 --output_c 25 \
#                 --n_memory 10 \
#                 --lambd 0.05 \
#                 --lr 5e-5 \
#                 --memory_initial True \
#                 --phase_type second_train \
#                 --ffn_ratio 1 \
#                 --patch_size 1 \
#                 --patch_stride 1 \
#                 --num_blocks 1 \
#                 --large_size 13 \
#                 --small_size 5 \
#                 --dims 128 \
#                 --head_dropout 0.0 \
#                 --dropout 0.1 \
#                 --pred_len 0\
#                 --backbone ModernTCN

# test
python main.py --anormly_ratio 1.0 \
                --num_epochs 10 \
                --batch_size 256  \
                --mode test \
                --dataset SMAP  \
                --data_path /media/media02/mhzheng/dataset/SMAP  \
                --input_c 25 \
                --output_c 25 \
                --n_memory 10 \
                --memory_initial False \
                --phase_type test \
                --ffn_ratio 1 \
                --patch_size 1 \
                --patch_stride 1 \
                --num_blocks 1 \
                --large_size 13 \
                --small_size 5 \
                --dims 128 \
                --head_dropout 0.0 \
                --dropout 0.1 \
                --pred_len 0\
                --backbone ModernTCN
