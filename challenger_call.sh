set -e

python scripts/CHALLENGER_call.py \
    --input examples/parquet_files/Toy_test.parquet \
    --weight models/CHALLENGER-LR  \
    --gpu 0 \
    --tokenizer-path data/challenger_tokenizer.json \
    --batch-size 32 \
    --output-dir outputs/ \
    --run-name toy \
    
    
