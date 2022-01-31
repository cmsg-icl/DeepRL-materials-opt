import numpy as np
import sys
import re
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
import itertools
import pickle

with open(r"bs_dict.pkl", "rb") as bs_dict_file:
    basis_sets = pickle.load(bs_dict_file)
    
currently_used_bs=['H_pob_DZVP_rev2','', 'Li_pob_DZVP_rev2', 'Be_pob_DZVP_rev2', 'B_pob_DZVP_rev2', 'C_pob_DZVP_rev2', 'N_pob_DZVP_rev2', 'O_pob_DZVP_rev2', 'F_pob_DZVP_rev2','Ne_6-31G', 'Na_pob_DZVP_rev2', 'Mg_pob_DZVP_rev2', 'Al_pob_DZVP_rev2', 'Si_pob_DZVP_rev2', 'P_pob_DZVP_rev2', 'S_pob_DZVP_rev2', 'Cl_pob_DZVP_rev2', 'Ar_6-31G','K_pob_DZVP_rev2', 'Ca_pob_DZVP_rev2', 'Sc_pob_DZVP_rev2', 'Ti_pob_DZVP_rev2', 'V_pob_DZVP_rev2', 'Cr_pob_DZVP_rev2', 'Mn_pob_DZVP_rev2', 'Fe_pob_DZVP_rev2', 'Co_pob_DZVP_rev2', 'Ni_pob_DZVP_rev2', 'Cu_pob_DZVP_rev2', 'Zn_pob_DZVP_rev2', 'Ga_pob_DZVP_rev2', 'Ge_pob_DZVP_rev2', 'As_pob_DZVP_rev2', 'Se_pob_DZVP_rev2', 'Br_pob_DZVP_rev2','', 'Rb_POB_DZVP_2018', 'Sr_POB_DZVP_2018','Y_POB_DZVP_2018', 'Zr_POB_DZVP_2018', 'Nb_POB_DZVP_2018','','Mo_POB_DZVP_2018', 'Ru_POB_DZVP_2018', 'Rh_POB_DZVP_2018', 'Pd_POB_DZVP_2018', 'Ag_POB_DZVP_2018', 'Cd_POB_DZVP_2018', 'In_POB_DZVP_2018', 'Sn_POB_DZVP_2018', 'Sb_POB_DZVP_2018', 'Te_POB_DZVP_2018', 'I_POB_DZVP_2018']
    
def find_elements(structure):
    ineq_elements=list(set(structure.species))
    ineq_Z=[elem.Z for elem in ineq_elements]
    return ineq_Z
    
class Crystal_input:
    
    def __init__(self,input_name,structure,shrink,FORCECALC=False):
        self.name = input_name
        self.structure = structure
        self.elements = find_elements(self.structure)
        self.shrink = shrink
        self.FORCECALC = FORCECALC
        
        
    def write_cry_gui(self,gui_file,dimensionality=3):

        with open(gui_file, 'w') as file: 

            atoms = SpacegroupAnalyzer(self.structure).get_primitive_standard_structure()

            #Is the structure symmetrysed?
            if 'SymmetrizedStructure' not in str(type(atoms)):
                atoms_symm = SpacegroupAnalyzer(atoms).get_symmetrized_structure()

            if dimensionality == 3:

                #First line 
                file.writelines('3   1   1\n')

                #Cell vectors
                for vector in atoms.lattice.matrix:
                    file.writelines(' '.join(str(n) for n in vector)+'\n')

                #N symm ops
                n_symmops = len(SpacegroupAnalyzer(atoms_symm).get_space_group_operations())
                file.writelines('{}\n'.format(str(n_symmops)))

                #symm ops
                for symmops in SpacegroupAnalyzer(atoms_symm).get_symmetry_operations(cartesian=True):  
                    file.writelines('{}\n'.format(' '.join(str(np.around(n,8)) for n in symmops.rotation_matrix[0])))
                    file.writelines('{}\n'.format(' '.join(str(np.around(n,8)) for n in symmops.rotation_matrix[1])))
                    file.writelines('{}\n'.format(' '.join(str(np.around(n,8)) for n in symmops.rotation_matrix[2])))
                    file.writelines('{}\n'.format(' '.join(str(np.around(n,8)) for n in symmops.translation_vector)))

            elif dimensionality == 2:
                file.writelines('2   1   1\n')
                #Cell vectors
                z_component = ['0.0000', '0.00000', '500.00000']
                for i,vector in enumerate(atoms.lattice.matrix[0:3,0:2]):
                    file.writelines(' '.join(str(n) for n in vector)+' '+z_component[i]+'\n')  
                #Center the slab
                #First center at z = 0.5
                atoms = center_slab(atoms)

                #Then center at z=0.0
                translation = np.array([0.0, 0.0, -0.5])
                atoms.translate_sites(list(range(atoms.num_sites)),
                                             translation, to_unit_cell=False)

                #Remove symmops with z component
                sg = SpacegroupAnalyzer(atoms)
                ops = sg.get_symmetry_operations(cartesian=True)       
                symmops = []
                for op in ops:
                    if op.translation_vector[2] == 0.:
                        symmops.extend(op.rotation_matrix.tolist())
                        symmops.extend([op.translation_vector.tolist()])  

                #N symm ops
                n_symmops = int(len(symmops)/4)
                file.writelines('{}\n'.format(n_symmops))

                #symm ops
                for symmop in symmops:    
                    file.writelines('{}\n'.format(' '.join(str(np.around(n,8)) for n in symmop)))

            #N atoms
            file.writelines('{}\n'.format(atoms.num_sites))

            #atom number + coordinates cart
            for i in range(atoms.num_sites):
                atomic_number = atoms.atomic_numbers[i]
                atom_coord = ' '.join(str(np.around(n,5)) for n in atoms.cart_coords[i])
                file.writelines('{} {}\n'.format(atomic_number,atom_coord))

            #space group + n symm ops
            file.writelines('{} {}'.format(SpacegroupAnalyzer(atoms).get_space_group_number(),len(SpacegroupAnalyzer(atoms).get_space_group_operations())))

#     def pmg_structure_to_geom_inp(self):
#         geom_block=["CRYSTAL","0 0 0"]
#         space_group=SpacegroupAnalyzer(self.structure, symprec=0.01, angle_tolerance=0.2)
#         geom_block.append(str(space_group.get_space_group_number()))
#         std_conventional=space_group.get_conventional_standard_structure()
#         lat_params=[str(round(x,4)) if i>2 else str(round(x,3)) for i,x in enumerate(list(std_conventional.lattice.parameters))]
        
#         if space_group.get_lattice_type()=='cubic':
#             geom_block.append(lat_params[0])
#         elif space_group.get_lattice_type()=='hexagonal' or space_group.get_lattice_type()=='tetragonal':
#             geom_block.append(lat_params[0]+' '+lat_params[2])
#         elif space_group.get_lattice_type()=='trigonal':
#             geom_block.append(lat_params[0]+' '+lat_params[3])
#         elif space_group.get_lattice_type()=='orthorhombic':
#             geom_block.append(lat_params[0]+' '+lat_params[1]+' '+lat_params[2])
#         elif space_group.get_lattice_type()=='monoclinic':
#             geom_block.append(lat_params[0]+' '+lat_params[1]+' '+lat_params[2]+' '+lat_params[4])
#         elif space_group.get_lattice_type()=='triclinic':
#             geom_block.append(' '.join(x) for x in lat_params)
        
        
#         geom_block.append(str(len(self.sym_distinct_sites)))
#         for site in self.sym_distinct_sites:
#             geom_block.append(str(site.species.elements[0].Z)+" "+str(round(site.a,5))+" "+str(round(site.b,5))+" "+str(round(site.c,5)))
#         return geom_block

    def write_cry_input(self):
        self.write_cry_gui(self.name.replace(".d12",".gui"))
        geom_block= ["EXTERNAL"]
        func_block = ["OPTGEOM","PRINTFORCES","MAXCYCLE","1","END\n","END\n"] if self.FORCECALC==True else ["END\n"]
        bs_block = []
        for element in self.elements:
            bs_block += [s for s in basis_sets[currently_used_bs[element-1]].split("\r\n")]
        bs_block += ['99 0','ENDBS\n']
        dft_block = ["DFT","B3LYP","END\n"]
        scf_block = ["TOLINTEG","7 7 7 7 14","SHRINK",str(self.shrink)+" "+str(self.shrink)+" "+str(2*self.shrink),"END"]
        with open(self.name,'w') as file:
            cry_input_list=list(itertools.chain([self.name],geom_block,list(func_block),bs_block,dft_block,scf_block))
            cry_input=[x+'\n' if "END" not in x else x for x in cry_input_list]#does CRYSTAL care about an empty line at the end?
            for line in cry_input:
                file.writelines(line)

class Crystal_output:

    def __init__(self,output_name): 
       
        self.name = output_name
        #Check if the file exists
        try: 
            if output_name[-3:] != 'out' and  output_name[-4:] != 'outp':
                output_name = output_name+'.out'
            file = open(output_name, 'r')
            self.data = file.readlines()
            file.close()
        except:
            print('EXITING: a .out file needs to be specified')
            #dont really want exceptions, make if else out of this
            sys.exit(1)

        #Check the calculation converged
        self.converged = False
        self.requeu = True
        for i,line in enumerate(self.data[::-1]):
            if re.match(r' == SCF ENDED - CONVERGENCE ON ENERGY',line):
                self.converged = True
                self.requeu = False
                self.eoscf = len(self.data)-1-i
                self.energy =  float(line.split()[8])*27.2114 
                break
            elif re.match(r' == SCF ENDED - TOO MANY CYCLES',line): #add other conditions of not converged scf->reque here
                self.requeu=False
                break
            


    def band_gap(self):#,spin_pol=False):
        
        self.spin_pol = False
        for line in self.data:
            if re.match(r'^ SPIN POLARIZED',line):
                self.spin_pol = True
                break
                
  
        for i, line in enumerate(self.data[len(self.data)::-1]):
            if self.spin_pol == False:
                if re.match(r'^\s\w+\s\w+ BAND GAP',line):
                    self.band_gap = float(line.split()[4])
                    return self.band_gap
                elif re.match(r'^\s\w+ ENERGY BAND GAP',line):
                    self.band_gap = float(line.split()[4])
                    return self.band_gap
                elif re.match(r'^ POSSIBLY CONDUCTING STATE',line):
                    self.band_gap = False
                    return self.band_gap 
            else:
                #This might need some more work
                band_gap_spin = []
                if re.match(r'\s+ BETA \s+ ELECTRONS',line):
                    band_gap_spin.append(float(self.data[len(self.data)-i-3].split()[4]))
                    band_gap_spin.append(float(self.data[len(self.data)-i+3].split()[4]))
                    self.band_gap = np.array(band_gap_spin)
                    return self.band_gap
        if band_gap_spin == []:
            print('DEV WARNING: check this output and the band gap function in code_io')
                #elif re.match(r'^\s\w+ ENERGY BAND GAP',line1) != None:
                    #band_gap = [float(data[len(data)-i-j-7].split()[4]),float(line1.split()[4])]

    def get_forces(self):
        self.forces=[]
        self.lattice_gradient=[]
        i=self.eoscf+1
        while not re.match(r' CARTESIAN FORCES IN HARTREE/BOHR \(ANALYTICAL\)',self.data[i]) and i<len(self.data):
            i+=1
        i+=2
        while self.data[i]!='\n' and i<len(self.data):
            self.forces.append([float(x) for x in self.data[i].split()[2:]])
            i+=1
        while not re.match(r' GRADIENT WITH RESPECT TO THE CELL PARAMETER IN HARTREE/BOHR',self.data[i]) and i<len(self.data):
            i+=1
        i+=4
        while self.data[i]!='\n' and i<len(self.data):
            self.lattice_gradient.append([float(x) for x in self.data[i].split()])
            i+=1