## Setup
conda create --name lc0 python=3.10
pip install -r tf/requirements_tf_2_9.txt
pip install tensorboard
pip install tf-models-official

git clone https://github.com/LeelaChessZero/lczero-common.git

run init.sh in root

# Start training
python tf/train.py --cfg tf\configs\example.yaml --output /tmp/mymodel.txt

python tf/find_corrupted_files.py merged/

tensorboard --logdir leelalogs --port 6031 --host localhost

# Split data

Training start: 11.12.2025 10:20