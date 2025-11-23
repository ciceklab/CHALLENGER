#!/bin/bash
set -e

example_path="./examples"
cohort_name="Toy_test"

cd $example_path
# Create an array of sample names (remove the .cram extension)
samples=(*.cram)
echo "Found CRAM files: ${samples[@]}"

for cram_file in "${samples[@]}"; do

    # Extract sample name (remove .cram extension)
    sample_name="${cram_file%.cram}"
    echo $cram_file
    echo $sample_name
    echo $example_path
    mkdir -p read_depths

    echo "=========================================="
    echo "Processing sample: $sample_name"
    echo "=========================================="

    echo "Step 1: Calculating read depth with samtools..."
    samtools depth -H -b ../data/hg38_genesLocationWider.bed \
        "$cram_file" \
        -o "read_depths/${sample_name}.txt"


    echo "Step 2: Running AveragePoolingReadDepthForGenes_wider.py..."
    python ../scripts/AveragePoolingReadDepthForGenes_wider.py \
        -s "$sample_name" \
        -t ../data/hg38_genesInfoWider.txt \
        -r "read_depths/" \
        -o "read_depths/"

    echo "Step 3: Removing temporary depth file..."
    rm "read_depths/${sample_name}.txt"

    echo "Finished sample: $sample_name"
    echo
done

echo "=========================================="
echo "Step 4: Creating cohort parquet file..."
echo "=========================================="
mkdir -p parquet_files

python ../scripts/Generate_parquet_file.py \
    -i read_depths/ \
    -g ../data/hg38_gene_index_lookup.npy \
    -t ../data/hg38_genesInfo.txt \
    -o parquet_files \
    -c ${cohort_name}

echo "Pipeline completed successfully!"
