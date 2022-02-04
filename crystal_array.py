import subprocess
import numpy as np
import json
from pathlib import Path
from crystal_helper import *

iteration = int(np.loadtxt("./iteration.txt"))
chunk_size = 2

with open ("./ComputeDir/array_script.qsub") as f:
    array_script_lines=f.readlines()

if iteration==0:
    with open('./structures.json','r') as f:
        database_query=json.load(f)[0:4]
else:
    with open('./RL_results/batch'+str(iteration-1)+'.json') as f:
        database_query=json.load(f)

chunked_database=[database_query[i:i+chunk_size] for i in range(0,len(database_query),chunk_size)]

for i,chunk in enumerate(chunked_database):
    p=Path("./ComputeDir/batch"+str(iteration)+"/chunk"+str(i))
    p.mkdir(parents=True,exist_ok=True)
    chunk_array_script=array_script_lines
    for structure in chunk:
        #write crystal input in chunk folder
        pmg_structure_obj=Structure.from_dict(structure[1])
        cry_input=Crystal_input(str(p)+"/"+structure[0]+".d12",pmg_structure_obj)
        cry_input.write_cry_input()
    
    chunk_array_script[4]=chunk_array_script[4].replace("1-2","1-"+str(len(chunk)))
    with open (str(p)+"/array_script.qsub","w") as f:
        f.writelines(chunk_array_script)
    subprocess.call("cd "+str(p)+";qsub array_script.qsub",shell=True)

    







