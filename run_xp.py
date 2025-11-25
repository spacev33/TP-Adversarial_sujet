import os

# Define the values of eps
eps_values = [1e-1,1e-2,1e-3]

# Loop through each value and execute the script
for eps in eps_values:
    print(f"Running evaluate.py with eps={eps}")
    command = f"python evaluate.py --path cnn --model cnn --attack fgsm --eps {eps}"
    os.system(command)
