import subprocess
import numpy as np
import json
from pathlib import Path
import shutil
from crystal_helper import *

iteration = int(np.loadtxt("./iteration.txt").split()[-1].strip())
chunk_size = 10

if iteration==0:
    with open('./structures.json','r') as f:
        database_query=json.load(f)
else:
    with open('./RL_results/batch'+str(iteration-1)+'.json') as f:
        database_query=json.load(f)

chunked_database=[database_query[i:i+chunk_size] for i in range(0,len(database_query),chunk_size)]

for i,chunk in enumerate(chunked_database):
    p=Path("./ComputeDir/batch"+str(iteration)+"/chunk"+str(i))
    p.mkdir(parents=True,exist_ok=True)
    for structure in chunk:
        #write crystal input in chunk folder
        cry_input=Crystal_input(str(p)+"/"+structure["material_id"]+".d12",structure["structure"],shrink=8)
        cry_input.write_cry_input()
    
    shutil.copy("./ComputeDir/array_script.qsub",str(p))
    subprocess.call("cd "+str(p);"qsub array_script.qsub",shell=True)

    







