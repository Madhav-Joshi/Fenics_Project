# read and write numpy arrays from npz (binary) files
import numpy as np

# Open the file to write in binary
outfile = open('./Heat_Conduction_Problem/TempFile','wb') # Or TempFile.npz is also allowed
x = np.arange(10)
y = np.sin(x)

np.savez(outfile, arr_x=x, arr_y=y)
outfile.close()

# Open the file to read in binary
outfile = open('./Heat_Conduction_Problem/TempFile','rb')
npzfile = np.load(outfile)

print(npzfile.files)
print(npzfile['arr_x'])
outfile.close()