OMP_NUM_THREADS=3
export OMP_NUM_THREADS

CUDA_VISIBLE_DEVICES=2  nohup python -u ./main.py -K 10  -B  32 -P 200 >> 10_32.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=3  nohup python -u ./main.py -K 13  -B  32 -P 200 >> 13_32.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=4  nohup python -u ./main.py -K 15  -B  32 -P 200 >> 15_32.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=5  nohup python -u ./main.py -K 18  -B  32 -P 200 >> 18_32.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=6  nohup python -u ./main.py -K 20  -B  32 -P 200 >> 20_32.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=7  nohup python -u ./main.py -K 23  -B  32 -P 200 >> 23_32.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=0  nohup python -u ./main.py -K 25  -B  32 -P 200 >> 25_32.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=1  nohup python -u ./main.py -K 28  -B  32 -P 200 >> 28_32.txt &

sleep 1s
CUDA_VISIBLE_DEVICES=2  nohup python -u ./main.py -K 25  -B  64 -P 200 >> 25_64.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=3  nohup python -u ./main.py -K 30  -B  64 -P 200 >> 30_64.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=4  nohup python -u ./main.py -K 35  -B  64 -P 200 >> 35_64.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=5  nohup python -u ./main.py -K 40  -B  64 -P 200 >> 40_64.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=6  nohup python -u ./main.py -K 45  -B  64 -P 200 >> 45_64.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=7  nohup python -u ./main.py -K 50  -B  64 -P 200 >> 50_64.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=0  nohup python -u ./main.py -K 55  -B  64 -P 200 >> 55_64.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=1  nohup python -u ./main.py -K 60  -B  64 -P 200 >> 60_64.txt &

sleep 1s
CUDA_VISIBLE_DEVICES=2  nohup python -u ./main.py -K 50  -B 128 -P 200 >> 50_128.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=3  nohup python -u ./main.py -K 60  -B 128 -P 200 >> 60_128.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=4  nohup python -u ./main.py -K 70  -B 128 -P 200 >> 70_128.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=5  nohup python -u ./main.py -K 80  -B 128 -P 200 >> 80_128.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=6  nohup python -u ./main.py -K 90  -B 128 -P 200 >> 90_128.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=7  nohup python -u ./main.py -K 100 -B 128 -P 200 >> 100_128.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=0  nohup python -u ./main.py -K 110 -B 128 -P 200 >> 110_128.txt &
sleep 1s
CUDA_VISIBLE_DEVICES=1  nohup python -u ./main.py -K 120 -B 128 -P 200 >> 120_128.txt &

# sleep 1s
# CUDA_VISIBLE_DEVICES=0  nohup python -u ./main.py -K 25  -B  64 -S 0 -P 200 >> 64_0.txt &
# CUDA_VISIBLE_DEVICES=1  nohup python -u ./main.py -K 30  -B  64 -S 0 -P 200 >> 64_0.txt &
# CUDA_VISIBLE_DEVICES=2  nohup python -u ./main.py -K 35  -B  64 -S 0 -P 200 >> 64_0.txt &
# CUDA_VISIBLE_DEVICES=3  nohup python -u ./main.py -K 45  -B  64 -S 0 -P 200 >> 64_0.txt &
# CUDA_VISIBLE_DEVICES=4  nohup python -u ./main.py -K 55  -B  64 -S 0 -P 200 >> 64_0.txt &

# CUDA_VISIBLE_DEVICES=5  nohup python -u ./main.py -K 25  -B  64 -S 1 -P 200 >> 64_1.txt &
# CUDA_VISIBLE_DEVICES=6  nohup python -u ./main.py -K 30  -B  64 -S 1 -P 200 >> 64_1.txt &
# CUDA_VISIBLE_DEVICES=7  nohup python -u ./main.py -K 35  -B  64 -S 1 -P 200 >> 64_1.txt &
# CUDA_VISIBLE_DEVICES=0  nohup python -u ./main.py -K 45  -B  64 -S 1 -P 200 >> 64_1.txt &
# CUDA_VISIBLE_DEVICES=1  nohup python -u ./main.py -K 55  -B  64 -S 1 -P 200 >> 64_1.txt &

# CUDA_VISIBLE_DEVICES=2  nohup python -u ./main.py -K 25  -B  64 -S 2 -P 200 >> 64_2.txt &
# CUDA_VISIBLE_DEVICES=3  nohup python -u ./main.py -K 30  -B  64 -S 2 -P 200 >> 64_2.txt &
# CUDA_VISIBLE_DEVICES=4  nohup python -u ./main.py -K 35  -B  64 -S 2 -P 200 >> 64_2.txt &
# CUDA_VISIBLE_DEVICES=6  nohup python -u ./main.py -K 45  -B  64 -S 2 -P 200 >> 64_2.txt &
# CUDA_VISIBLE_DEVICES=7  nohup python -u ./main.py -K 55  -B  64 -S 2 -P 200 >> 64_2.txt &

# CUDA_VISIBLE_DEVICES=0  nohup python -u ./main.py -K 25  -B  64 -S 3 -P 200 >> 64_3.txt &
# CUDA_VISIBLE_DEVICES=1  nohup python -u ./main.py -K 30  -B  64 -S 3 -P 200 >> 64_3.txt &
# CUDA_VISIBLE_DEVICES=2  nohup python -u ./main.py -K 35  -B  64 -S 3 -P 200 >> 64_3.txt &
# CUDA_VISIBLE_DEVICES=3  nohup python -u ./main.py -K 45  -B  64 -S 3 -P 200 >> 64_3.txt &
# CUDA_VISIBLE_DEVICES=4  nohup python -u ./main.py -K 55  -B  64 -S 3 -P 200 >> 64_3.txt &

# CUDA_VISIBLE_DEVICES=5  nohup python -u ./main.py -K 25  -B  64 -S 4 -P 200 >> 64_4.txt &
# CUDA_VISIBLE_DEVICES=6  nohup python -u ./main.py -K 30  -B  64 -S 4 -P 200 >> 64_4.txt &
# CUDA_VISIBLE_DEVICES=7  nohup python -u ./main.py -K 35  -B  64 -S 4 -P 200 >> 64_4.txt &
# CUDA_VISIBLE_DEVICES=7  nohup python -u ./main.py -K 45  -B  64 -S 4 -P 200 >> 64_4.txt &
# CUDA_VISIBLE_DEVICES=6  nohup python -u ./main.py -K 55  -B  64 -S 4 -P 200 >> 64_4.txt &
