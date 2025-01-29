


# Activate virtual environment for the first part of the pipeline
echo "Activating virtual environment..."
source ~/miniconda3/etc/profile.d/conda.sh

conda activate taxposed

# Train the model
echo "Training the model..."
python train_taxposed.py