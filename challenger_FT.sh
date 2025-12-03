set -e

python scripts/CHALLENGER_FT.py \
    --input examples/parquet_files/Toy_FT.parquet \
    --init-weight models/CHALLENGER-LR  \
    --gpu 0 \
    --num-epoch 100 \
    --tokenizer-path data/challenger_tokenizer.json \
    --batch-size 32 \
    --output-dir FT_weights/ \
    --run-name toy \
    
    
