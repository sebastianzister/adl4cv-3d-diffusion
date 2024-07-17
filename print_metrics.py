import re
import argparse

def main(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract the required values using regex
    nn_cd_acc = re.search(r"'1-NN-CD-acc': ([\d\.]+)", content).group(1)
    nn_emd_acc = re.search(r"'1-NN-EMD-acc': ([\d\.]+)", content).group(1)
    lgan_cov_cd = re.search(r"'lgan_cov-CD': ([\d\.]+)", content).group(1)
    lgan_cov_emd = re.search(r"'lgan_cov-EMD': ([\d\.]+)", content).group(1)
    lgan_mmd_cd = re.search(r"'lgan_mmd-CD': ([\d\.]+)", content).group(1)
    lgan_mmd_emd = re.search(r"'lgan_mmd-EMD': ([\d\.]+)", content).group(1)

    # Convert extracted values to float
    nn_cd_acc = float(nn_cd_acc)
    nn_emd_acc = float(nn_emd_acc)
    lgan_cov_cd = float(lgan_cov_cd)
    lgan_cov_emd = float(lgan_cov_emd)
    lgan_mmd_cd = float(lgan_mmd_cd)
    lgan_mmd_emd = float(lgan_mmd_emd)

    # Calculate the scaled values
    nn_cd_acc_scaled = nn_cd_acc * 100
    nn_emd_acc_scaled = nn_emd_acc * 100
    lgan_cov_cd_scaled = lgan_cov_cd * 100
    lgan_cov_emd_scaled = lgan_cov_emd * 100
    lgan_mmd_cd_scaled = lgan_mmd_cd * 1000
    lgan_mmd_emd_scaled = lgan_mmd_emd * 100

    # Print the scaled values
    print(f'1-NN-CD-acc * 100: {nn_cd_acc_scaled}')
    print(f'1-NN-EMD-acc * 100: {nn_emd_acc_scaled}')
    print(f'lgan_cov-CD * 100: {lgan_cov_cd_scaled}')
    print(f'lgan_cov-EMD * 100: {lgan_cov_emd_scaled}')
    print(f'lgan_mmd-CD * 1000: {lgan_mmd_cd_scaled}')
    print(f'lgan_mmd-EMD * 100: {lgan_mmd_emd_scaled}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print scaled metrics from file.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the file containing metrics')
    args = parser.parse_args()
    main(args.file)