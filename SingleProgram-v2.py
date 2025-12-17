import os
import json
import glob
import numpy as np
from datetime import datetime as dt 
from zoneinfo import ZoneInfo
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
total_samples = len(interleaved_IQ)

if total_samples % 2 == 1:
    interleaved_IQ = interleaved_IQ[:-1]
else:
    pass

total_samples = len(interleaved_IQ)

I = interleaved_IQ[0::2]
Q = interleaved_IQ[1::2]

total_complex_samples = len(I)

datetime_object = dt.now(ZoneInfo("Asia/Kolkata"))
datetime_stamp = datetime_object.strftime("%Y-%m-%d %H:%M:%S %z")[:-2] + ":" + "" + datetime_object.strftime("%Y-%m-%d %H:%M:%S%z")[-2:]


n = 5
print(f'\n Below are {n} I and Q data samples \n')
print(I[:n])          # I samples are printed here
print(Q[:n])          # Q samples are printed here


Fs = float(input("\nWhat's the Sampling Frequency (in MHz)? \n"))
fc = float(input("\nEnter the center frequency (in GHz) \n"))
name = str(input("\nWho is working with data? \n"))
description = str(input("\nShort description of the capture \n"))

switch = int(input("\nWhat is the type of transmitted signal? \nFor CW --> Type 1  \nPulsed RF --> Type 2 \n\n"))

if switch == 1:
    print(f''' \nCW Signal can be divided and framed based on time chunks. The signal in the loaded file is {round(total_samples/(2*Fs*1e6), 4)} seconds long and has {total_complex_samples} I and Q samples each''')
    delta_t = float(input("\nWhat is the desired length of each data chunk (in seconds)? \n" ))
    #no_samples = int(Fs*1e6*delta_t)
    interleavedIQ_chunk_size = 2 * int(Fs * 1e6 * delta_t)
    I_chunk_size = int(interleavedIQ_chunk_size/2)
    num_chunks = len(interleaved_IQ) // interleavedIQ_chunk_size       # number of samples in a single chunk
    #print(f'\nNo. of samples acquired for {delta_t} seconds chunk is: {no_samples} \n')
    print(f'\nNo. of complex samples acquired for {delta_t} seconds chunk is: {int(I_chunk_size)} \n')
    
elif switch == 2:
    print('\nThe signal can be framed based on pulses')    
    ipp = float(input("\nWhat's the IPP of the Tx pulse (in micro-seconds)? \n"))
    nci = int(input("\nNo. of Coherent Integrations \n"))
    nfft = int(input("\nNo. of FFT points \n"))
    interleavedIQ_chunk_size = 2*int(ipp * nci * Fs)  # chunk size in terms of number of samples(complex )
    I_chunk_size = int(interleavedIQ_chunk_size/2)
    num_chunks = len(interleaved_IQ) // interleavedIQ_chunk_size       # number of samples in a single chunk
    print(f'\nAccording to the set parameters, we can divide the fed data file into {num_chunks} chunks. \
          \nEach chunk has {I_chunk_size} complex samples.')

else:
    print('Enter a valid input')
    


# Chunking the data to parts for ease of accessing only a part of data
chunks = np.array_split(interleaved_IQ[:num_chunks * interleavedIQ_chunk_size], num_chunks)
# IQ[:num_chunks * chunk_size] --- this command trims the array data to the no. of elements = num_chunks*interleavedIQ_chunk_size
# Now that we have a single array data with so many elements, now we have to cut the data into chunks
# np.array_split(IQ[:num_chunks * interleavedIQ_chunk_size], num_chunks) --- this command splits the data in num_chunks




buf_arr = np.zeros(interleavedIQ_chunk_size)

# Path for .bin files
split_directory_path = file_path.split('\\')[:-1] # Source file's parent Directory Path string split to form a List
directory_path = '\\'.join(split_directory_path) # Source file's parent Directory Path formed to a string
bin_directory_path = os.path.join(directory_path, 'Bin Files') # Path to store .bin and.json files 
mat_directory_path = os.path.join(directory_path, 'Mat Files') # Path to store .bin and.json files 

desired_no_chunks = int(input(f"Out of {num_chunks}, how many chunks do you want to convert to .mat files? \n"))

for i in range(desired_no_chunks):
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

if switch == 1:
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
                "core:Sampling_Rate_MHz": Fs,
                "core:Center_Frequency_GHz": fc,
                "core:datatype": get_data_type_str(bin_data),
                "core:Transmitted_Waveform": "Continuous Wave",
                "core:Description": description,
                "core:Author": name,
                
                
                #"core:datatype": "cf32_le", # Complex float, 32-bit, little-endian
                
                
            },
            "captures": [
                {
                    "core:Labelling_Date_&_Time": datetime_stamp,
                    #"core:datetime": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "core:sample_start": "0",
                }
            ],
            "annotations": [
                {
                    "core:comment": "An example annotation on a specific segment",
                    "core:frame_number": f'frame_{count}',
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

elif switch == 2:
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
                "core:Sampling_Rate_MHz": Fs,
                "core:Center_Frequency_GHz": fc,
                "core:datatype": get_data_type_str(bin_data),
                "core:Transmitted_Waveform": "Pulsed RF",
                "core:Description": description,
                "core:Author": name,
                
                
                #"core:datatype": "cf32_le", # Complex float, 32-bit, little-endian
                
                
            },
            "captures": [
                {
                    "core:Labelling_Date_&_Time": datetime_stamp,
                    #"core:datetime": dt.datetime.now(dt.timezone.utc).isoformat(),
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

else:
    print('Enter a valid switch value')


print('\n############\nHeader or Metadata files for each Data file are created \n############')

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

print(f'\n############\nConverted {total} (.bin + .json) file pairs to .mat files \n############')


