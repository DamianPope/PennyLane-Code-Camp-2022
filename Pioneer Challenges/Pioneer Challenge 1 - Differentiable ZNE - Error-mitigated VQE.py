# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 08:52:12 2022

@author: dpope
"""

import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

def hydrogen_hamiltonian(d):
    """Creates the H_2 Hamiltonian from a separation distance.

    Args:
        d (float): The distance between a hydrogen atom and the hydrogen molecule's centre of mass.

    Returns:
        H (qml.Hamiltonian): The H_2 Hamiltonian.
        qubits (int): The number of qubits needed to simulate the H_2 Hamiltonian.
    """

    symbols = symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -d, 0.0, 0.0, d])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

    return H, qubits

def ansatz_template(param, wires):
    """The unitaries used for creating an ansatz for subsequent VQE calculations.

    Args:
        param (np.array): A single differentiable parameter
        wires (list(int)): A list of wires that the unitaries are applied to.
    """
    qml.BasisState([1, 1, 0, 0], wires=wires)
    qml.DoubleExcitation(param, wires=wires)


def VQE(qnode):    
    """Performs a VQE routine given a QNode.

    Args:
        qnode (qml.QNode):
            The ansatz that will be optimized in order to find the ground state
            of molecular hydrogen.

    Retuns:
        final_energy (float): The final energy from the VQE routine.
    """
    
    param = np.array(0.0, requires_grad=True)
    num_iters = 20
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    for _ in range(num_iters):
        param = opt.step(qnode, param)
       
    final_energy = qnode(param)

    return final_energy

#implements the VQE algorithm for all the noisy circuits
def VQE_noisy(folded_cost_fn):    

    """
    Args:
        A cost function that has a quantum circuit that's a folded circuit

    Retuns:
        final_energy (float): The final energy from the VQE routine.
    """

    param = np.array(0.0, requires_grad=True)
    num_iters = 20
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    for _ in range(num_iters):
        param = opt.step(folded_cost_fn, param)
       
    final_energy = folded_cost_fn(param)

    return final_energy

def qnode_ansatzes(d, scale_factors):
    """Generates ideal and mitigated qnodes.

    Args:
        d (float): The distance between a hydrogen atom and the hydrogen molecule's centre of mass.
        scale_factors (list(int)): A list of scale factors used for ZNE.

    Returns:
       qnode_ideal (qml.QNode): The ideal QNode (no noise).
       qnodies_mitgated (list(qml.QNode)): A list of QNodes that are mitigated. len(qnodes_mitigated) = len(scale_factors).
    """
    H, qubits = hydrogen_hamiltonian(d)

    noise_gate = qml.DepolarizingChannel
    noise_strength = 0.05

    # Put your code here #
    dev_ideal = qml.device("default.mixed", wires=qubits)

    def cost_function(param):
        ansatz_template(param,range(qubits))
        return qml.expval(H)
        
    #create the QNode for the *ideal* VQE circuit
    qnode_ideal = qml.QNode(cost_function, dev_ideal)

    qnodes_mitigated = []
    #A list that stores the folded quantum circuits created by using the global folding protocol with different scale factors
    folded_circuits_list = []    

    #create & initialize a dictionary as an empty dictionary
    globals_dict = {}

    #
    #create qnodes_mitigated 
    #
    for i in range(len(scale_factors)):
        
        deviceNameString = "noisy_dev" + str(i)
           
        #create a device that will be a noisy device
        globals_dict[deviceNameString] = qml.device("default.mixed", wires=qubits)
        
        #add noise to device
        globals_dict[deviceNameString] = qml.transforms.insert(noise_gate, noise_strength,position="all")(globals_dict[deviceNameString])
        
        #create a QNODE associated with the noisy device
        qnodeNameString = "noisy_qnode" + str(i)
        globals_dict[qnodeNameString] = qml.QNode(cost_function, globals_dict[deviceNameString])
        
        #create a folded circuit with a scale factor of scale_factors[i]
        folded_circuit = qml.transforms.fold_global(globals_dict[qnodeNameString], scale_factors[i])
      
        #add folded circuit to a list of folded circuits
        folded_circuits_list.append(folded_circuit)   

        qnodes_mitigated.append(globals_dict[qnodeNameString])
        
    return qnode_ideal, qnodes_mitigated,folded_circuits_list

def extrapolation(d, scale_factors, plot=False):
    """Performs ZNE to obtain a zero-noise estimate on the ground state energy of H_2.

    Args:
        d (float): The distance between a hydrogen atom and the hydrogen molecule's centre of mass.
        scale_factors (list(int)): A list of scale factors used for ZNE.

    Returns:
        ideal_energy (float): The ideal energy from a noise-less VQE routine.
        zne_energy (float): The zero-noise estimate of the ground state energy of H_2.

        These two energies are returned in that order within a numpy array.

    """

    qnode_ideal, qnodes_mitigated,folded_circuits_list = qnode_ansatzes(d, scale_factors)

    ideal_energy = np.round_(VQE(qnode_ideal), decimals=6)
    
    #calculate the mitigated energies using the "standard" method of implementing VQE given a cost function (i.e., folded_circ) that's a quantum circuit    
    #Note that the output of the cost function is the average energy of the final state.
    mitigated_energies = [VQE_noisy(folded_circ) for folded_circ in folded_circuits_list]
    
    # Put your code here #
    #Create the quadratic polynomial that best fits the mitigated energies 
    energy_polynomial = np.polyfit(scale_factors,mitigated_energies,2)

    #polyval evaluates a polynomial at a specific value of the dependent variable (in this case, the scale factor)
    zne_energy= np.polyval(energy_polynomial,0)

    return np.array([ideal_energy, zne_energy]).tolist()

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    d = json.loads(test_case_input)
    scale_factors = [1, 2, 3]
    energies = extrapolation(d, scale_factors)
    return str(energies)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-2
    ), "Your extrapolation isn't quite right!"


test_cases = [['0.6614', '[-1.13619, -0.41168]']]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")