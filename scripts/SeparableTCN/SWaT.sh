# # SWaT

python main.py --anomaly_ratio 0.5 \
                --num_epochs 20 \
                --batch_size 512 \
                --mode train \
                --dataset SWaT \
                --data_path /media/media02/mhzheng/dataset/SWaT \
                --input_c 51 \
                --output_c 51 \
                --n_memory 51 \
                --lambd 0.005 \
                --lr 5e-4 \
                --memory_initial False \
                --phase_type None \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32\
                --head_dropout 0.1 \
                --dropout 0.1 \
                --backbone SeparableTCN

python main.py --anomaly_ratio 0.5 \
                --num_epochs 20  \
                --batch_size 512  \
                --mode memory_initial \
                --dataset SWaT  \
                --data_path /media/media02/mhzheng/dataset/SWaT  \
                --input_c 51 \
                --output_c 51 \
                --n_memory 51 \
                --lambd 0.1\
                --lr 1e-4 \
                --memory_initial True \
                --phase_type second_train \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.1 \
                --dropout 0.1 \
                --backbone SeparableTCN

# test
python main.py --anomaly_ratio 0.5 \
                --num_epochs 10   \
                --batch_size 512  \
                --mode test \
                --dataset SWaT  \
                --data_path /media/media02/mhzheng/dataset/SWaT  \
                --input_c 51 \
                --output_c 51 \
                --n_memory 51 \
                --lr 1e-6 \
                --memory_initial False \
                --phase_type test \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.1 \
                --dropout 0.1 \
                --backbone SeparableTCN