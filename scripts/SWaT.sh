# SWaT

python main.py --anomaly_ratio 0.1 \
                --num_epochs 200 \
                --batch_size 256 \
                --mode train \
                --dataset SWaT \
                --data_path /media/media02/mhzheng/dataset/SWaT \
                --input_c 51 \
                --output_c 51 \
                --n_memory 10 \
                --lambd 0.005 \
                --lr 1e-4 \
                --memory_initial False \
                --phase_type None \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 3 \
                --large_size 51 \
                --small_size 5 \
                --dims 128 \
                --head_dropout 0.0 \
                --dropout 0.1 

python main.py --anomaly_ratio 0.1 \
                --num_epochs 200  \
                --batch_size 256  \
                --mode memory_initial \
                --dataset SWaT  \
                --data_path /media/media02/mhzheng/dataset/SWaT  \
                --input_c 51 \
                --output_c 51 \
                --n_memory 10 \
                --lambd 0.05 \
                --lr 5e-5 \
                --memory_initial True \
                --phase_type second_train \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 3 \
                --large_size 51 \
                --small_size 5 \
                --dims 128 \
                --head_dropout 0.0 \
                --dropout 0.1 

# test
python main.py --anomaly_ratio 0.1 \
                --num_epochs 100   \
                --batch_size 256  \
                --mode test \
                --dataset SWaT  \
                --data_path /media/media02/mhzheng/dataset/SWaT  \
                --input_c 51 \
                --output_c 51 \
                --n_memory 10 \
                --memory_initial False \
                --phase_type test \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 3 \
                --large_size 51 \
                --small_size 5 \
                --dims 128 \
                --head_dropout 0.0 \
                --dropout 0.1 