pip install -e .
cd melo

create a virtual environment
python -m venv venv
source venv/bin/activate

create a metadata file
python split_and_transcribe.py


python preprocess_text.py --metadata data/metadata.list

bash train.sh data/config.json 4

python infer.py --text "hello world" -m /path/to/checkpoint/G_<iter>.pth -o <output_dir>