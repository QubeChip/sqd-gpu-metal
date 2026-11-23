"""Minimal H2 full-space selected-CI solve using the native Metal core."""

import numpy as np
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci import direct_spin1, selected_ci

from sqd_gpu_metal import (
    MetalSelectedCIContractor, absorb_h1e, davidson_lowest, make_hdiag, mixed_rdm_q60,
)


def main():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    if not mf.converged:
        raise RuntimeError("Hartree-Fock did not converge")

    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    strings = tuple(fci.cistring.make_strings(range(norb), n) for n in nelec)
    links = selected_ci._all_linkstr_index(strings, norb, nelec)

    import mlx.core as mx

    contractor = MetalSelectedCIContractor(absorb_h1e(h1e, eri, nelec), links, nelec)
    diagonal = make_hdiag(
        mx.array(h1e.astype(np.float32)), mx.array(eri.astype(np.float32)), strings
    ).reshape(-1)
    shape = tuple(map(len, strings))
    solved = davidson_lowest(
        lambda vector: contractor.contract(vector.reshape(shape)).reshape(-1),
        diagonal,
        tolerance=1e-5,
        max_cycles=100,
        max_space=20,
    )
    if not solved.converged:
        raise RuntimeError(f"Metal Davidson did not converge: {solved.residual_norm:.3e}")

    reference_energy, _ = selected_ci.kernel_fixed_space(
        selected_ci.SelectedCI(), h1e, eri, norb, nelec, strings,
        tol=1e-12, max_cycle=100, max_space=20, tol_residual=1e-10,
    )
    coefficients = np.asarray(solved.coefficients, dtype=np.float64).reshape(shape)
    coefficients /= np.linalg.norm(coefficients)
    vector = selected_ci._as_SCIvector(coefficients, strings)
    image = selected_ci.contract_2e(
        direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5), vector, norb, nelec
    )
    residual = float(np.linalg.norm(image - solved.energy * vector))
    tables = tuple(selected_ci.cre_des_linkstr(labels, norb, count)
                   for labels, count in zip(strings, nelec))
    mixed = mixed_rdm_q60(coefficients, norb, nelec, tables)
    reference_mixed = selected_ci.make_rdm2s(vector, norb, nelec)[1]
    mixed_error = float(np.max(np.abs(mixed - reference_mixed)))

    print(f"Metal electronic energy: {solved.energy:.12f} Ha")
    print(f"PySCF reference energy:  {reference_energy:.12f} Ha")
    print(f"Energy error:            {abs(solved.energy - reference_energy):.3e} Ha")
    print(f"Independent residual:    {residual:.3e}")
    print(f"Mixed 2-RDM max error:   {mixed_error:.3e}")
    print(f"Iterations / H@C calls:  {solved.iterations} / {solved.h_matvecs}")
    print(f"Native calls:            {contractor.native_calls}")
    print(f"Execution:               {contractor.execution}")
    if (abs(solved.energy - reference_energy) > 2e-5 or residual > 2e-5
            or mixed_error > 1e-11):
        raise RuntimeError("Metal result failed the independent PySCF check")


if __name__ == "__main__":
    main()
