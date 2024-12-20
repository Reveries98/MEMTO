# SMD
# python main.py --anomaly_ratio 0.5 --num_epochs 100 --batch_size 256 --mode train --dataset SMD --data_path /media/media02/mhzheng/dataset/SMD --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anomaly_ratio 0.5 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMD  --data_path /media/media02/mhzheng/dataset/SMD  --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# test
# python main.py --anomaly_ratio 0.5 --num_epochs 10  --batch_size 256 --mode test --dataset SMD  --data_path /media/media02/mhzheng/dataset/SMD  --input_c 38 --output_c 38 --n_memory 10 --memory_initial False --phase_type test


# SMAP
# python main.py --anomaly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset SMAP --data_path /media/media02/mhzheng/dataset/SMAP --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anomaly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMAP  --data_path /media/media02/mhzheng/dataset/SMAP  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anomaly_ratio 1.0 --num_epochs 10 --metrics PA   --batch_size 256  --mode test --dataset SMAP  --data_path /media/media02/mhzheng/dataset/SMAP  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# PSM
# python main.py --anomaly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset PSM --data_path /media/media02/mhzheng/dataset/PSM --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anomaly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset PSM  --data_path /media/media02/mhzheng/dataset/PSM  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# test
# python main.py --anomaly_ratio 1.0 --num_epochs 10  --batch_size 256  --mode test --dataset PSM  --data_path /media/media02/mhzheng/dataset/PSM  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# # MSL

# python main.py --anomaly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset MSL --data_path /media/media02/mhzheng/dataset/MSL --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anomaly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset MSL  --data_path /media/media02/mhzheng/dataset/MSL  --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anomaly_ratio 1.0 --num_epochs 10    --batch_size 256  --mode test --dataset MSL  --data_path /media/media02/mhzheng/dataset/MSL  --input_c 55 --output_c 55 --n_memory 10 --memory_initial False --phase_type test


# SWaT

# python main.py --anomaly_ratio 0.1 --num_epochs 100 --batch_size 256 --mode train --dataset SWaT --data_path /media/media02/mhzheng/dataset/SWaT --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anomaly_ratio 0.1 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SWaT  --data_path /media/media02/mhzheng/dataset/SWaT  --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# test
python main.py --anomaly_ratio 0.1 --num_epochs 100   --batch_size 256  --mode test --dataset SWaT  --data_path /media/media02/mhzheng/dataset/SWaT  --input_c 51 --output_c 51 --n_memory 10 --memory_initial False --phase_type test