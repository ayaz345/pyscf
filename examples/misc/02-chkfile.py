#!/usr/bin/env python

'''
This example shows how to access the data stored in checkpoint file,
Also how to quickly update an object using the data from the checkpoint file.
'''


import numpy
import h5py
from pyscf import gto, scf, ci
from pyscf.lib import chkfile

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol)
mf.chkfile = 'example.chk'
mf.run()
print(f'E(HF) = {mf.e_tot}')

scf_result_dic = chkfile.load('example.chk', 'scf')
mf_new = scf.RHF(mol)
mf_new.__dict__.update(scf_result_dic)
print(f'E(HF) from chkfile = {mf.e_tot}')

myci = ci.CISD(mf).run()
myci.dump_chk()
print(f'E(CISD) = {myci.e_tot}')

cisd_result_dic = chkfile.load('example.chk', 'cisd')
myci_new = ci.CISD(mf_new)
myci_new.__dict__.update(cisd_result_dic)
print(f'E(CISD) from chkfile = {myci_new.e_tot}')

mol_new = chkfile.load_mol('example.chk')
print(numpy.allclose(mol.atom_coords(), mol_new.atom_coords()))
print(numpy.allclose(mol.atom_charges(), mol_new.atom_charges()))


with h5py.File('example.chk') as f:
    print('\nCheckpoint file is a HDF5 file. data are stored in file/directory structure.')
    print('/', f.keys())
    print('/scf', f['scf'].keys())
    print('/scf/mo_occ', f['scf/mo_occ'].value)
    print('/cisd', f['cisd'].keys())
    print('\nMolecular object (mol) is seriealized to json format and stored')
    print(f"/mol: {f['mol'].value[:20]} ...")
