docker run --platform linux/arm64 -it --rm \
    -v "$(pwd)/src":/app/src \
    -v "$(pwd)/input_data":/app/input_data \
    -v "$(pwd)/output":/app/output \
    ryouheishimizuhata/ela_for_md:1.0 \
    conda run -n elaenv python /app/src/execute.py --target_path /app/input_data --save_path /app/output
