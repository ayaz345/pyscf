#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import h5py
from pyscf import gto, scf, ao2mo

'''
Save the transformed integrals in the given file in HDF5 format
'''

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
)

myhf = scf.RHF(mol)
myhf.kernel()

orb = myhf.mo_coeff
ftmp = tempfile.NamedTemporaryFile()
print(f'MO integrals are saved in file  {ftmp.name}  under dataset "eri_mo"')
ao2mo.kernel(mol, orb, ftmp.name)

# Read integrals via h5py module
with h5py.File(ftmp.name) as f:
    eri_4fold = f['eri_mo']
    print(
        f'MO integrals (ij|kl) with 4-fold symmetry i>=j, k>=l have shape {str(eri_4fold.shape)}'
    )

