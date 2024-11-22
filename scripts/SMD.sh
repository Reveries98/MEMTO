# SMD
python main.py --anormly_ratio 0.5 \
                --num_epochs 200 \
                --batch_size 256 \
                --mode train \
                --dataset SMD \
                --data_path /media/media02/mhzheng/dataset/SMD \
                --input_c 38 \
                --output_c 38 \
                --n_memory 10 \
                --lambd 0.005 \
                --lr 1e-4 \
                --memory_initial False \
                --phase_type None \
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.0 \
                --dropout 0.1 \

python main.py --anormly_ratio 0.5 \
                --num_epochs 200  \
                --batch_size 256  \
                --mode memory_initial \
                --dataset SMD  \
                --data_path /media/media02/mhzheng/dataset/SMD  \
                --input_c 38 \
                --output_c 38 \
                --n_memory 10 \
                --lambd 0.05 \
                --lr 5e-5 \
                --memory_initial True \
                --phase_type second_train\
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.0 \
                --dropout 0.1 \

# test
python main.py --anormly_ratio 0.5 \
                --num_epochs 10  \
                --batch_size 256 \
                --mode test \
                --dataset SMD  \
                --data_path /media/media02/mhzheng/dataset/SMD  \
                --input_c 38 \
                --output_c 38 \
                --n_memory 10 \
                --memory_initial False \
                --phase_type test\
                --ffn_ratio 1 \
                --patch_size 8 \
                --patch_stride 4 \
                --num_blocks 1 \
                --large_size 71 \
                --small_size 5 \
                --dims 32 \
                --head_dropout 0.0 \
                --dropout 0.1 \