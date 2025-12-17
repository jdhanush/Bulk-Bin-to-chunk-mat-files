import os
import json
import glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import spectrogram
from scipy.io import savemat#, loadmat 
from sigmf.utils import get_data_type_str
#########################################################################################################
# First Program
#########################################################################################################
# Source file
file_path = r'C:\Users\user\Desktop\Dhanush Files\SDR Files\experiments\9. Integration\Data\small_file.bin'
interleaved_IQ = np.fromfile(file_path, dtype = np.float32)
I = interleaved_IQ[0::2]
Q = interleaved_IQ[1::2]

n = 5
print(f'\n Below are {n} I and Q data samples \n')
print(I[:5])          # I samples are printed here
print(Q[:5])          # Q samples are printed here


Fs = int(input("\nWhat's the Sampling Frequency in Hz? \n"))
fc = int(input("\nEnter the center freq in Hz \n"))
ipp = int(input("\nWhat's the IPP of the Tx pulse in micro-seconds? \n"))
nci = int(input("\nNo. of Coherent Integrations \n"))
nfft = int(input("\nNo. of FFT points \n"))
name = str(input("\nWho is working with data? \n"))
description = str(input("\nShort description of the capture \n"))
Ts = round(1/Fs, 4)

chunk_size = 2*int(1*ipp * 1e-6 * nci * Fs)  # chunk size in terms of number of samples(complex )
num_chunks = len(interleaved_IQ) // chunk_size       # number of samples in a single chunk

# Chunking the data to parts for ease of accessing only a part of data
chunks = np.array_split(interleaved_IQ[:num_chunks * chunk_size], num_chunks)
# IQ[:num_chunks * chunk_size] --- this command trims the array data to the no. of elements = num_chunks*chunk_size
# Now that we have a single array data with so many elements, now we have to cut the data into chunks
# np.array_split(IQ[:num_chunks * chunk_size], num_chunks) --- this command splits the data in num_chunks


print(f'\nAcc. to the set parameters, we can divide the fed data file into {num_chunks} chunks. \
      \nEach chunk has {chunk_size} complex samples')

buf_arr = np.zeros(chunk_size)

# Path for .bin files
split_directory_path = file_path.split('\\')[:-1] # Source file's parent Directory Path string split to form a list
directory_path = '\\'.join(split_directory_path) # Source file's parent Directory Path formed to a string
bin_directory_path = os.path.join(directory_path, 'Bin Files') # Path to store .bin and.json files 
mat_directory_path = os.path.join(directory_path, 'Mat Files') # Path to store .bin and.json files 

for i in range(num_chunks//8):
    buf_arr = chunks[i]
    buf_arr.tofile(os.path.join(bin_directory_path, f'chunk_{i}.bin'))
    #buf_arr.tofile(rf'C:\Users\user\Desktop\Dhanush Files\SDR Files\experiments\9. Integration\Bin files\chunk_{i}.bin') # extension is optional
    print(f'File chunk_{i} saved in .bin format')
    
print('############\nFinished 1st Program: Data Chunking activity Done! \n############')    
#########################################################################################################
# Second Program
#########################################################################################################    

# Extracts the numeric index from a predictable filename of the form
# Example: 'signal_12.bin' -> split -> ['signal', '12.bin'] -> 12
def get_index(filename):
    base = os.path.splitext(filename)[0]     # remove ".bin"
    index_str = base.split("_")[-1]          # take the part after last "_"
    return int(index_str)                    # convert to integer


# # Path for .bin files
# split_directory_path = file_path.split('\\')[:-1]
# directory_path = '\\'.join(split_directory_path)
# bin_directory_path = os.path.join(directory_path, 'Bin Files')

# Grab only .bin files from the bin_directory_path
bin_files = [
    f for f in os.listdir(bin_directory_path)
    if f.lower().endswith(".bin")
    ]

# Sort files based on extracted index
sorted_bin_files = sorted(bin_files, key=get_index)

print(f'\n############\n{len(bin_files)} Bin Files Identified Independently! \n############\n')


# For each .bin/data file, we write a header/metadata file
count = 0
for bin_source_file in sorted_bin_files:
    count += 1
    binfile_path = os.path.join(bin_directory_path, bin_source_file) # joins and builds the total data path
    bin_data = np.fromfile(binfile_path) # loads the data from the .bin file
    #bin_source_file = binfile_path.split('\\')[-1] 
    
    # Define the SigMF metadata as a Python dictionary
    metadata = {
        "global": {
            "core:Source_File": bin_source_file,
            "core:Sampling_Rate_Hz": Fs,
            "core:Center_Frequency_Hz": fc,
            "core:datatype": get_data_type_str(bin_data),
            "core:Description": description,
            "core:Author": name,
            
            
            #"core:datatype": "cf32_le", # Complex float, 32-bit, little-endian
            
            
        },
        "captures": [
            {
                "core:datetime": dt.datetime.now(dt.timezone.utc).isoformat(),
                "core:sample_start": "0",
            }
        ],
        "annotations": [
            {
                "core:comment": "An example annotation on a specific segment",
                "core:frame_number": f'frame_{count}',
                "core:ipp": ipp,
                "core:nci": nci,
                "core:nfft": nfft,
                "core:sample_count": 200,
                "core:sample_start": 0
                }
            ]
    }
    
    
    plain_file = bin_source_file.split('.')[0]
    json_filename = str([plain_file + '.json'][0])
    jsonfile_path = os.path.join(bin_directory_path, json_filename)
    
    with open(jsonfile_path, "w") as f:
        json.dump(metadata, f, indent=4)
     
    print(f"Successfully created {json_filename} file")


print(f'\n############\nHeader or Metadata files for each Data file are created \n############')

#########################################################################################################
# Third Program
#########################################################################################################    
file_list = glob.glob(os.path.join(bin_directory_path, "*.bin"))
total = len(file_list)

if total == 0:
    print("No .bin files found.")
    

os.makedirs(mat_directory_path, exist_ok=True)

for idx, filepath in enumerate(file_list):
    try:
        with open(filepath, "rb") as handle:
            iq_seq = np.fromfile(handle, dtype=np.float32)
        
        json_path = filepath.replace(".bin", ".json")

        with open(json_path, "r") as jf:
            metadata = json.load(jf)

        out_file = os.path.join(
            mat_directory_path,
            os.path.basename(filepath).replace(".bin", ".mat")
        )

        savemat(out_file, {
            "iq-data": iq_seq,
            "metadata": json.dumps(metadata)
        })

        print(f"Saved: {out_file}")

    except Exception as e:
        print(f"Error: {filepath}: {e}")

    print(f"Progress: {(idx+1)/total*100:.2f}%")

print(f'\n############\nDone converting {total} .bin and .json file pairs to .mat files \n############')



