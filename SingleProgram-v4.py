import os
import sys
import json
import glob
import shutil
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from datetime import datetime as dt 
from scipy.signal import spectrogram
from scipy.io import savemat #, loadmat 
import matplotlib.animation as animation
from sigmf.utils import get_data_type_str


#########################################################################################################
# First Program
#########################################################################################################
# Source .bin file
source_file_path = r'C:\Users\user\Desktop\Dhanush Files\SDR Files\experiments\9. Integration\Data\small_file.bin'
interleaved_IQ = np.fromfile(source_file_path, dtype = np.float32)
     
# this if-else makes sure to maintain even number of interleaved samples to be distributed between I and Q
if len(interleaved_IQ) % 2 == 1:
    interleaved_IQ = interleaved_IQ[:-1]
else:
    pass


total_samples = len(interleaved_IQ)  # No. of Interleaved Samples

# Splitting I and Q from the interleaved data
I = interleaved_IQ[0::2]        
Q = interleaved_IQ[1::2]

total_complex_samples = len(I)  # No. of complex samples

# datetime_object = dt.now(ZoneInfo("Asia/Kolkata"))
# datetime_stamp = datetime_object.strftime("%Y-%m-%d %H:%M:%S %z")[:-2] + ":" + "" + datetime_object.strftime("%Y-%m-%d %H:%M:%S%z")[-2:]

n = 5
print(f'\n Below are {n} I and Q data samples \n')
print(I[:n])          # I samples are printed here
print(Q[:n])          # Q samples are printed here

#########################################################################################################
# Taking User Inputs
#########################################################################################################
Fs = float(input("\nWhat's the Sampling Frequency (in MHz)? \n"))
fc = float(input("\nEnter the center frequency (in GHz) \n"))
nrgb = int(input("\nNo. of Rangebins \n"))
nfft = int(input("\nNo. of FFT points \n"))
now = dt.now()
year    = now.year
month   = now.month
day     = now.day
hour    = now.hour
minute  = now.minute
second  = now.second

switch = int(input("\nWhat is the type of transmitted signal? \nFor CW --> Type [1]  \nPulsed RF --> Type [2] \n"))
#########################################################################################################
# Counting the no. of chunks in the signal file and no. of samples per chunk
#########################################################################################################
if switch == 1:
    cw_pul = 1 # This tells that the tx'ed signal is Continuous wave
    print(f''' \nCW The signal can be framed based on time chunks. \nThe signal in the loaded file is {round(total_samples/(2*Fs*1e6), 4)} seconds long and has {total_complex_samples} I and Q samples each''')
    delta_t = float(input("\nWhat is the desired length of each data chunk (in seconds)? \n" ))
    interleavedIQ_chunk_size = 2 * int(Fs * 1e6 * delta_t) # For a given sampling freq., we get both I and Q samples, in this eqn., we are picking both IQ samples 
    I_chunk_size = int(interleavedIQ_chunk_size/2) # No. of I or Q or IQ samples 
    num_chunks = len(interleaved_IQ) // interleavedIQ_chunk_size  # No. of chunks in the provided signal
    print(f'\nNo. of complex samples acquired for {delta_t} seconds chunk is: {int(I_chunk_size)} \n')
    
elif switch == 2:
    cw_pul = 2 # This tells that the tx'ed signal is Pulsed RF
    print('\nThe signal can be framed based on pulses')    
    ipp = float(input("\nWhat's the IPP of the Tx pulse (in micro-seconds)? \n"))
    print(f'\nFor the given IPP, there are {int(total_complex_samples/(ipp*Fs))} pulses in total in the loaded signal file')
    nci = int(input("\nNo. of Coherent Integrations \n"))
    interleavedIQ_chunk_size = 2*int(ipp * nci * Fs)  # chunk size in terms of number of samples(complex )
    I_chunk_size = int(interleavedIQ_chunk_size/2)      # No. of I or Q or IQ samples 
    num_chunks = len(interleaved_IQ) // interleavedIQ_chunk_size       # No. of chunks in the provided signal
    print(f'\nAccording to the set parameters, we can divide the fed data file into {num_chunks} chunks. \
          \nEach chunk has {I_chunk_size} complex samples.')
          
else:
    print('Enter a valid input')
    print("\nExiting the program")    
    sys.exit(0)


# Chunking the data to parts for ease of accessing only a part of data
chunks = np.array_split(interleaved_IQ[:num_chunks * interleavedIQ_chunk_size], num_chunks)
# IQ[:num_chunks * chunk_size] --- this command trims the array data to the no. of elements = num_chunks*interleavedIQ_chunk_size
# Now that we have a single array data with so many elements, now we have to cut the data into chunks
# np.array_split(IQ[:num_chunks * interleavedIQ_chunk_size], num_chunks) --- this command splits the data in num_chunks

buf_arr = np.zeros(interleavedIQ_chunk_size)        # An empty array of size equal to the chunk size

#########################################################################################################
# Creating new Mat and Bin files while clearing the old contents
#########################################################################################################
# Building a new path for .bin and .mat files
split_directory_path = source_file_path.split('\\')[:-1] # Source file's parent Directory Path string split to form a List
directory_path = '\\'.join(split_directory_path) # Source file's parent Directory Path formed to a string
bin_directory_path = os.path.join(directory_path, 'Bin Files') # Path to store .bin and.json files 
mat_directory_path = os.path.join(directory_path, 'Mat Files') # Path to store .mat files 

# Creates Bin directory or leaves as is, if it exists already
bin_out_dir = Path(bin_directory_path)   
bin_out_dir.mkdir(parents=True, exist_ok=True)

# Creates Mat directory or leaves as is, if it exists already
mat_out_dir = Path(mat_directory_path)
mat_out_dir.mkdir(parents=True, exist_ok=True)


desired_no_chunks = int(input(f"\nOut of {num_chunks}, how many chunks do you want to convert to .mat files? \n"))

for item in bin_out_dir.iterdir():
    if item.is_file() or item.is_symlink():
        item.unlink()          # delete file or symlink
    elif item.is_dir():
        shutil.rmtree(item)    # delete directory and its contents
print("\nClearing all the old content from:", bin_directory_path)

for item in mat_out_dir.iterdir():
    if item.is_file() or item.is_symlink():
        item.unlink()          # delete file or symlink
    elif item.is_dir():
        shutil.rmtree(item)    # delete directory and its contents
print("Clearing all the old content from:", mat_directory_path, '\n')


for i in range(desired_no_chunks):
    buf_arr = chunks[i]
    buf_arr.tofile(os.path.join(bin_directory_path, f'chunk_{i}.bin'))
    print(f'chunk_{i}.bin file saved successfully')
    
print('\n############\nFinished 1st Program: Data Chunking activity Done! \n############')    

#########################################################################################################
# Creating Metadata(.json file) for each .bin file
#########################################################################################################    
# Extracts the numeric index from a predictable filename of the form
# Example: 'signal_12.bin' -> split -> ['signal', '12.bin'] -> 12
def get_index(filename):
    base = os.path.splitext(filename)[0]     # remove ".bin"
    index_str = base.split("_")[-1]          # take the part after last "_"
    return int(index_str)                    # convert to integer

# Grab only .bin files from the bin_directory_path
bin_files = [
    f for f in os.listdir(bin_directory_path)
    if f.lower().endswith(".bin")
    ]

print(f'\n############\n{len(bin_files)} Bin Files Identified Independently! \n############\n')

# Sort files based on extracted index
sorted_bin_files = sorted(bin_files, key=get_index)

if switch == 1:
    # For each .bin/data file, we write a header/metadata file
    count = 0
    for bin_source_file in sorted_bin_files:
        count += 1
        binfile_path = os.path.join(bin_directory_path, bin_source_file) # joins and builds the total data path
        bin_data = np.fromfile(binfile_path) # loads the data from the .bin file
        
        # Defining the SigMF metadata as a Python list
        ipp = 0        # ipp = 0 for CW signals
        metadata = [Fs, fc, count, ipp, nrgb, nfft, cw_pul, year, month, day, hour, minute, second]
        plain_file = bin_source_file.split('.')[0]
        json_filename = str([plain_file + '.json'][0])
        jsonfile_path = os.path.join(bin_directory_path, json_filename)
        
        f = open(jsonfile_path, "w")
        json.dump(metadata, f, indent=1)
        f.close()  # easy to forget = bugs
               
        print(f"{json_filename} file created successfully")

elif switch == 2:
    # For each .bin/data file, we write a header/metadata file
    count = 0
    for bin_source_file in sorted_bin_files:
        count += 1
        binfile_path = os.path.join(bin_directory_path, bin_source_file) # joins and builds the total data path
        bin_data = np.fromfile(binfile_path) # loads the data from the .bin file
        
        # Defining the SigMF metadata as a Python list
        metadata = [Fs, fc, count, ipp, nrgb, nfft, cw_pul, year, month, day, hour, minute, second]
        plain_file = bin_source_file.split('.')[0]
        json_filename = str([plain_file + '.json'][0])
        jsonfile_path = os.path.join(bin_directory_path, json_filename)
        
        f = open(jsonfile_path, "w")
        json.dump(metadata, f, indent=1)
        f.close()  # easy to forget = bugs
        
        # with open(jsonfile_path, "w") as f:
        #    json.dump(metadata, f, indent=4)
         
        print(f"{json_filename} file created successfully")

print('\n############\nHeader or Metadata files corresponding to each Data file are created \n############ \n\n')

#########################################################################################################
# Third Program, Merging .bin and .json files to form .mat files
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

        out_file = os.path.join(mat_directory_path, os.path.basename(filepath).replace(".bin", ".mat"))

        savemat(out_file, {
            "metadata": json.dumps(metadata),
            "data": iq_seq
        })

        print(f"Saved: {out_file}")

    except Exception as e:
        print(f"Error: {filepath}: {e}")

    print(f"Progress: {(idx+1)/total*100:.2f}%")

print(f'\n############\nConverted {total} (.bin + .json) file pairs to .mat files \n############')

#########################################################################################################
# Option to clear .bin files
#########################################################################################################    
clear_switch = str(input('''\nDo you want to clear Bin files and its contents? 
                         \nClear Bin file while retaining the Mat file  --> Type [1] \nRetain both Mat and Bin files --> Type [2] \n'''))
                         
if clear_switch == '1':
    shutil.rmtree(bin_directory_path)
    print('\nBin files directory cleared')
elif clear_switch == '2':
    print("\nBoth Bin and Mat files are retained !!!")
else:
    print("\nEnter a valid input")



























  