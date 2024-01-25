"""
Check files in simulation folder and delete incomplete subfolders (run locally!).

Created on Thu Jan 18 14:21:51 2024

@author: claireguepin
"""

import sys
import glob
import shutil

path_loc = '/Users/claireguepin/Projects/GRAND/GP300LibraryXi2023_iron/'
list_f = glob.glob(path_loc+'*')
print('Number of files = %i' % (len(list_f)))

for i in range(len(list_f)):
    list_files = glob.glob(list_f[i]+'/*')
    if len(list_files) != 3:
        print(list_f[i])
        # Remove directory if the 3 files have not been produced
        shutil.rmtree(list_f[i])
