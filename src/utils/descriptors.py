import math
from typing import List, Union

import numpy as np
import torch
from ase import Atoms
from dscribe import System
from dscribe.descriptors import CoulombMatrix, EwaldSumMatrix, SineMatrix
from torch_geometric.data import Data
from torch_sparse import SparseTensor

class TrueEwaldSumMatrix(EwaldSumMatrix):
    def create_single(
        self,
        system: Union[Atoms, System],
        accuracy=1e-5,
        w=1,
        rcut=None,
        gcut=None,
        a=None,
    ) -> np.ndarray:
        """
        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            accuracy (float): The accuracy to which the sum is converged to.
                Corresponds to the variable :math:`A` in
                https://doi.org/10.1080/08927022.2013.840898. Used only if
                gcut, rcut and a have not been specified. Provide either one
                value or a list of values for each system.
            w (float): Weight parameter that represents the relative
                computational expense of calculating a term in real and
                reciprocal space. This has little effect on the total energy,
                but may influence speed of computation in large systems. Note
                that this parameter is used only when the cutoffs and a are set
                to None. Provide either one value or a list of values for each
                system.
            rcut (float): Real space cutoff radius dictating how many terms are
                used in the real space sum. Provide either one value or a list
                of values for each system.
            gcut (float): Reciprocal space cutoff radius. Provide either one
                value or a list of values for each system.
            a (float): The screening parameter that controls the width of the
                Gaussians. If not provided, a default value of :math:`\\alpha =
                \sqrt{\pi}\left(\\frac{N}{V^2}\\right)^{1/6}` is used.
                Corresponds to the standard deviation of the Gaussians. Provide
                either one value or a list of values for each system.
        Returns:
            ndarray: The matrix either as a 2D array or as
                a 1D array depending on the setting self._flatten.
        """
        self.q = system.get_atomic_numbers()
        self.q_squared = self.q ** 2
        self.n_atoms = len(system)
        self.volume = system.get_volume()
        self.sqrt_pi = math.sqrt(np.pi)

        # If a is not provided, use a default value
        if a is None:
            a = (self.n_atoms * w / (self.volume ** 2)) ** (
                1 / 6
            ) * self.sqrt_pi

        # If the real space cutoff, reciprocal space cutoff and a have not been
        # specified, use the accuracy and the weighting w to determine default
        # similarly as in https://doi.org/10.1080/08927022.2013.840898
        if rcut is None and gcut is None:
            f = np.sqrt(-np.log(accuracy))
            rcut = f / a
            gcut = 2 * a * f
        elif rcut is None or gcut is None:
            raise ValueError(
                "If you do not want to use the default cutoffs, please provide "
                "both cutoffs rcut and gcut."
            )

        self.a = a
        self.a_squared = self.a ** 2
        self.gcut = gcut
        self.rcut = rcut

        # Transform the input system into the internal System-object
        system = self.get_system(system)

        matrix = self.get_matrix(system)

        return matrix


def add_ewaldsum_features(
    data_list: List[Data],
    accuracy=1e-5,
    w=1,
    rcut=None,
    gcut=None,
    a=None,
    n_jobs=1,
    only_physical_cores=False,
    verbose=False,
) -> List[Data]:
    """Add the EwaldSum matrix for the given data of a homogeneous graph.
    Args:
        data_list (list of :class:`torch_geometric.data.Data`): One or
            many data graph.
        accuracy (float): The accuracy to which the sum is converged to.
            Corresponds to the variable :math:`A` in
            https://doi.org/10.1080/08927022.2013.840898. Used only if
            gcut, rcut and a have not been specified. Provide either one
            value or a list of values for each system.
        w (float): Weight parameter that represents the relative
            computational expense of calculating a term in real and
            reciprocal space. This has little effect on the total energy,
            but may influence speed of computation in large systems. Note
            that this parameter is used only when the cutoffs and a are set
            to None. Provide either one value or a list of values for each
            system.
        rcut (float): Real space cutoff radius dictating how many terms are
            used in the real space sum. Provide either one value or a list
            of values for each system.
        gcut (float): Reciprocal space cutoff radius. Provide either one
            value or a list of values for each system.
        a (float): The screening parameter that controls the width of the
            Gaussians. If not provided, a default value of :math:`\\alpha =
            \sqrt{\pi}\left(\\frac{N}{V^2}\\right)^{1/6}` is used.
            Corresponds to the standard deviation of the Gaussians. Provide
            either one value or a list of values for each system.
        n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
            the calculation across samples. Defaults to serial calculation
            with n_jobs=1. If a negative number is given, the used cpus
            will be calculated with, n_cpus + n_jobs, where n_cpus is the
            amount of CPUs as reported by the OS. With only_physical_cores
            you can control which types of CPUs are counted in n_cpus.
        only_physical_cores (bool): If a negative n_jobs is given,
            determines which types of CPUs are used in calculating the
            number of jobs. If set to False (default), also virtual CPUs
            are counted.  If set to True, only physical CPUs are counted.
        verbose(bool): Controls whether to print the progress of each job
            into to the console.
    Returns:
        torch_geometric.data: List of updated data object with Sine matrix `.ewaldsum` for the given datas.
    """

    assert all(
        hasattr(data, "atomic_numbers")
        and hasattr(data, "pos")
        and hasattr(data, "cell")
        for data in data_list
    )

    esm = TrueEwaldSumMatrix(n_atoms_max=1, flatten=False, permutation="none")
    atoms_list = [
        Atoms(
            positions=data.pos,
            numbers=data.atomic_numbers,
            cell=data.cell[0],
            pbc=True,
        )
        for data in data_list
    ]

    # Combine input arguments
    n_samples = len(atoms_list)
    if np.ndim(accuracy) == 0:
        accuracy = n_samples * [accuracy]
    if np.ndim(w) == 0:
        w = n_samples * [w]
    if np.ndim(rcut) == 0:
        rcut = n_samples * [rcut]
    if np.ndim(gcut) == 0:
        gcut = n_samples * [gcut]
    if np.ndim(a) == 0:
        a = n_samples * [a]
    inp = [
        (i_sys, i_accuracy, i_w, i_rcut, i_gcut, i_a)
        for i_sys, i_accuracy, i_w, i_rcut, i_gcut, i_a in zip(
            atoms_list, accuracy, w, rcut, gcut, a
        )
    ]

    
    if n_jobs > 1:
        esm_m_list = esm.create_parallel(
            inp,
            esm.create_single,
            n_jobs,
            only_physical_cores=only_physical_cores,
            verbose=verbose,
            prefer="threads",
        )
    else:
        esm_m_list = []
        for arg in inp:
            esm_m_list.append(esm.create_single(*arg))

    for data, ewaldsum_matrix in zip(data_list, esm_m_list):
        data.ewaldsum = SparseTensor.from_dense(torch.tensor(ewaldsum_matrix))

    return data_list


class TrueCoulombMatrix(CoulombMatrix):
    def create_single(self, system: Union[Atoms, System]) -> np.ndarray:
        """
        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
        Returns:
            ndarray: The matrix either as a 2D array or as
                a 1D array depending on the setting self._flatten.
        """
        # Transform the input system into the internal System-object
        system = self.get_system(system)

        matrix = self.get_matrix(system)

        return matrix


class TrueSineMatrix(SineMatrix):
    def create_single(self, system: Union[Atoms, System]) -> np.ndarray:
        """
        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
        Returns:
            ndarray: The matrix either as a 2D array or as
                a 1D array depending on the setting self._flatten.
        """
        # Transform the input system into the internal System-object
        system = self.get_system(system)

        matrix = self.get_matrix(system)

        return matrix


def add_sine_features(
    data_list: List[Data],
    n_jobs: int = 1,
    only_physical_cores: bool = False,
    verbose: bool = False,
) -> List[Data]:
    """Add the Sine matrix for the given data of a homogeneous graph.
    Args:
        data_list (list of :class:`torch_geometric.data.Data`): One or
            many data graph.
        n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
            the calculation across samples. Defaults to serial calculation
            with n_jobs=1. If a negative number is given, the used cpus
            will be calculated with, n_cpus + n_jobs, where n_cpus is the
            amount of CPUs as reported by the OS. With only_physical_cores
            you can control which types of CPUs are counted in n_cpus.
        only_physical_cores (bool): If a negative n_jobs is given,
            determines which types of CPUs are used in calculating the
            number of jobs. If set to False (default), also virtual CPUs
            are counted.  If set to True, only physical CPUs are counted.
        verbose(bool): Controls whether to print the progress of each job
            into to the console.
    Returns:
        torch_geometric.data: List of updated data object with Sine matrix `.sine` for the given datas.
    """

    assert all(
        hasattr(data, "atomic_numbers")
        and hasattr(data, "pos")
        and hasattr(data, "cell")
        for data in data_list
    )

    sm = TrueSineMatrix(n_atoms_max=1, flatten=False, permutation="none")
    atoms_list = [
        Atoms(
            positions=data.pos,
            numbers=data.atomic_numbers,
            cell=data.cell[0],
            pbc=True,
        )
        for data in data_list
    ]

    # Combine input arguments
    inp = [(i_sys,) for i_sys in atoms_list]
    if n_jobs > 1:
        sm_m_list = sm.create_parallel(
            inp,
            sm.create_single,
            n_jobs,
            only_physical_cores=only_physical_cores,
            verbose=verbose,
            prefer="threads",
        )
    else:
        sm_m_list = []
        for arg in inp:
            sm_m_list.append(sm.create_single(*arg))

    for data, sine_matrix in zip(data_list, sm_m_list):
        data.sine = SparseTensor.from_dense(torch.tensor(sine_matrix))

    return data_list


def add_coulomb_features(
    data_list: List[Data],
    n_jobs: int = 1,
    only_physical_cores: bool = False,
    verbose: bool = False,
) -> List[Data]:
    """Add the Coulomb matrix for the given data of a homogeneous graph.
    Args:
        data_list (list of :class:`torch_geometric.data.Data`): One or
            many data graph.
        n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
            the calculation across samples. Defaults to serial calculation
            with n_jobs=1. If a negative number is given, the used cpus
            will be calculated with, n_cpus + n_jobs, where n_cpus is the
            amount of CPUs as reported by the OS. With only_physical_cores
            you can control which types of CPUs are counted in n_cpus.
        only_physical_cores (bool): If a negative n_jobs is given,
            determines which types of CPUs are used in calculating the
            number of jobs. If set to False (default), also virtual CPUs
            are counted.  If set to True, only physical CPUs are counted.
        verbose(bool): Controls whether to print the progress of each job
            into to the console.
    Returns:
        torch_geometric.data: List of updated data object with Coulomb matrix `.coulomb` for the given datas.
    """

    assert all(
        hasattr(data, "atomic_numbers")
        and hasattr(data, "pos")
        and hasattr(data, "cell")
        for data in data_list
    )

    cm = TrueCoulombMatrix(n_atoms_max=1, flatten=False, permutation="none")
    atoms_list = [
        Atoms(
            positions=data.pos,
            numbers=data.atomic_numbers,
            cell=data.cell[0],
            pbc=True,
        )
        for data in data_list
    ]

    # Combine input arguments
    inp = [(i_sys,) for i_sys in atoms_list]
    if n_jobs > 1:
        cm_m_list = cm.create_parallel(
            inp,
            cm.create_single,
            n_jobs,
            only_physical_cores=only_physical_cores,
            verbose=verbose,
            prefer="threads",
        )
    else:
        cm_m_list = []
        for arg in inp:
            cm_m_list.append(cm.create_single(*arg))
    
    for data, coulomb_matrix in zip(data_list, cm_m_list):
        data.coulomb = SparseTensor.from_dense(torch.tensor(coulomb_matrix))

    return data_list
