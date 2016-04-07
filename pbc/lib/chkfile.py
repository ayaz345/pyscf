#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import json
import h5py
import pyscf.pbc.gto
import pyscf.lib.chkfile
from pyscf.lib.chkfile import load_chkfile_key, load
from pyscf.lib.chkfile import dump_chkfile_key, dump, save

def load_cell(chkfile):
    '''Load Cell object from chkfile.
    The save_cell/load_cell operation can be used a serialization method for Cell object.
    
    Args:
        chkfile : str
            Name of chkfile.

    Returns:
        An (initialized/built) Cell object

    Examples:

    >>> from pyscf.pbc import gto, scf
    >>> cell = gto.Cell()
    >>> cell.build(atom='He 0 0 0')
    >>> scf.chkfile.save_cell(cell, 'He.chk')
    >>> scf.chkfile.load_cell('He.chk')
    <pyscf.pbc.gto.cell.Cell object at 0x7fdcd94d7f50>
    '''
    with h5py.File(chkfile, 'r') as fh5:
        try:
            cell = pyscf.pbc.gto.loads(fh5['mol'].value)
        except:
            from numpy import array  # for eval() function
            celldic = eval(fh5['mol'].value)
            cell = pyscf.pbc.gto.cell.unpack(celldic)
            cell.build(False, False)
    return cell

dump_cell = save_cell = pyscf.lib.chkfile.save_mol

