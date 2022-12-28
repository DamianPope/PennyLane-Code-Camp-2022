# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:54:30 2022

@author: dpope
"""

import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

#from functools import partial
#from pennylane.fourier import coefficients

#

def fourier_squared_distance(list_of_coeffs, param_list):
    """
    Returns the squared l2-distance in Fourier space between a function
    characterized by its Fourier coefficients and the output of the
    quantum model

    Args:
        list_of coeffs (list(float)): A list of seven coefficients
                                      corresponding to the Fourier
                                      coefficients of the function we
                                      want to approximate
        param_list (list(float)): A list of six parameters characterizing
                                  the angles in the trainable circuit.

    Returns: (float): Squared l2-distance between the given function
                      and the output of the quantum model
    """

    dev = qml.device("default.qubit", wires=3)

    # Feel free to define any helper functions, such as subcircuits, here.

    def W(angles):
        for i in range(3):
            qml.RX(angles[i],wires=i)
    
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[1,2])
        qml.CNOT(wires=[2,0])
        
    @qml.qnode(dev)
    def circuit(param_list, x):
        """This circuit returns the PauliZ expectation of
        the quantum model in the statement"""
        # Put your code here #
        
        W(param_list[0:3])
        
        for i in range(3):
            qml.RX(x,wires=i)
        
        W(param_list[3:6])
        
        return qml.expval(qml.PauliZ(0))

    # Write a function that calculates the squared l2-distance here
    L2 = 0.0
    
    partial_circuit = functools.partial(circuit, param_list)
    
    #NOTE: 
    #the argument 1 = the number of input variables in the partial_circuit function 
    #the argument 3 = maximum number of Fourier coefficients that are calculated = 2*d + 1,
    #where d is a parameter that corresponds to the maximum frequency that we include in our Fourier decomposition
    quantumCoeffs = qml.fourier.coefficients(partial_circuit,1,3)
    
    for i in range(len(list_of_coeffs)):
        L2 += (abs(list_of_coeffs[i]-quantumCoeffs[i]))**2
        
    # Return your final answer here
    return L2

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = fourier_squared_distance(*ins)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    """
    Compare solution with expected.

    Args:
            solution_output: The output from an evaluated solution. Will be
            the same type as returned.
            expected_output: The correct result for the test case.

    Raises:
            ``AssertionError`` if the solution output is incorrect in any way.
    """

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-2
    ), "Your calculated squared distance isn't quite right."


test_cases = [['[[-1.12422548e-01,  0.0, 9.47909940e-02, 0.0, 0.0, 9.47909940e-02, 0.0],[2,2,2,3,4,5]]', '0.0036766085933034303'], ['[[-2.51161988e-01, 0.0, 1.22546112e-01, 0.0, 0.0,  1.22546112e-01, 0.0],[1.1,0.3,0.4,0.6,0.8,0.9]]', '0.6538589174369286']]

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