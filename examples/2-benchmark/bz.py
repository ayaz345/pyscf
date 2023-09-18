#!/usr/bin/env python
import pyscf
from pyscf.tools.mo_mapping import mo_comps
from benchmarking_utils import setup_logger, get_cpu_timings

log = setup_logger()

for bas in ('3-21g', '6-31g**', 'cc-pVTZ', 'ANO-Roos-TZ'):
    mol = pyscf.M(atom = '''
c   1.217739890298750 -0.703062453466927  0.000000000000000
h   2.172991468538160 -1.254577209307266  0.000000000000000
c   1.217739890298750  0.703062453466927  0.000000000000000
h   2.172991468538160  1.254577209307266  0.000000000000000
c   0.000000000000000  1.406124906933854  0.000000000000000
h   0.000000000000000  2.509154418614532  0.000000000000000
c  -1.217739890298750  0.703062453466927  0.000000000000000
h  -2.172991468538160  1.254577209307266  0.000000000000000
c  -1.217739890298750 -0.703062453466927  0.000000000000000
h  -2.172991468538160 -1.254577209307266  0.000000000000000
c   0.000000000000000 -1.406124906933854  0.000000000000000
h   0.000000000000000 -2.509154418614532  0.000000000000000
''',
                  basis = bas)
    cpu0 = get_cpu_timings()

    mf = mol.RHF().run()
    cpu0 = log.timer(f'C6H6 {bas} RHF', *cpu0)

    mymp2 = mf.MP2().run()
    cpu0 = log.timer(f'C6H6 {bas} MP2', *cpu0)

    mymc = mf.CASSCF(6, 6)
    idx_2pz = mo_comps('2pz', mol, mf.mo_coeff).argsort()[-6:]
    mo = mymc.sort_mo(idx_2pz, base=0)
    mymc.kernel(mo)
    cpu0 = log.timer(f'C6H6 {bas} CASSCF', *cpu0)

    mycc = mf.CCSD().run()
    cpu0 = log.timer(f'C6H6 {bas} CCSD', *cpu0)

    mf = mol.RKS().run(xc='b3lyp')
    cpu0 = log.timer(f'C6H6 {bas} B3LYP', *cpu0)

    mf = mf.density_fit().run()
    cpu0 = log.timer(f'C6H6 {bas} density-fit RHF', *cpu0)

