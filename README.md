# ELA_for_Microbiome_data
The scripts for running energy landscape using micro biome dataset.

# How to run
・excute.py is the main script to run energy landscape. <br>
・run the following command:
```bash:
python execute.py
```

# Run docker image
・At first, clone this repository as file and move to the file's directry
```bash:
git clone https://github.com/ryouheishimizuhata/ELA_for_Microbiome_data
cd ../ELA_for_Microbiome_data
```

・pull the docker image from ryouheishimizuhata/ela_for_md:1.0 .
```bash:
docker pull ryouheishimizuhata/ela_for_md:1.0
```
・activate virtual environment in container
```bash:
docker run --platform linux/arm64 -it --rm \
    -v "$(pwd)/src":/app/src \
    -v "$(pwd)/input_data":/app/input_data \
    -v "$(pwd)/output":/app/output \
    ryouheishimizuhata/ela_for_md:1.0 /bin/bash
```
・ After activate bash 
```bash:
conda activate elaenv
```
・execute main file
```bash:
python /app/src/execute.py --target_path /app/input_data --save_path /app/output
```
