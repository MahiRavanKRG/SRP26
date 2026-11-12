import os

def create_slurm_scripts():
    datasets = ['ETTh1.csv', 'Weather.csv', 'Traffic.csv']
    models = ['dlinear', 'freq', 'wavelet']
    pred_lens = [96, 192, 336, 720]
    seq_len = 336
    
    os.makedirs('slurm_scripts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    script_count = 0
    
    for dataset in datasets:
        dataset_name = dataset.split('.')[0]
        for model in models:
            for pred_len in pred_lens:
                job_name = f"{model}_{dataset_name}_{pred_len}"
                script_path = f"slurm_scripts/run_{job_name}.sh"
                
                content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.log
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# Ensure you activate your uv virtual environment before submitting, 
# or uncomment and modify the line below:
# source /path/to/your/venv/bin/activate

srun python experiment.py --dataset {dataset} --model {model} --seq_len {seq_len} --pred_len {pred_len} --epochs 10 --batch_size 32
"""
                with open(script_path, 'w') as f:
                    f.write(content)
                script_count += 1
                
    # Create a master submit script
    with open("slurm_scripts/submit_all.sh", "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("for script in run_*.sh; do\n")
        f.write("    sbatch \"$script\"\n")
        f.write("done\n")
        
    print(f"Successfully generated {script_count} SLURM job scripts in 'slurm_scripts/' directory.")
    print("Run 'bash slurm_scripts/submit_all.sh' on the cluster to queue all experiments.")

if __name__ == '__main__':
    create_slurm_scripts()
