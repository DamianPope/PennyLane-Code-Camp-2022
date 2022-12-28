# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:51:19 2022

@author: dpope
"""

import functools
import json
import math
import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

def circuit():
    """
    Succession of gates that will generate the requested matrix.
    This function does not receive any arguments nor does it return any values.
    """

    # Put your solution here ...
    # You only have to put U3 or CNOT gates
    
    #The decomposition below of a CCZ gate comes from: 
    #https://www.researchgate.net/figure/Circuit-diagram-decomposition-of-the-three-qubit-ccz-gate-in-terms-of-seven-single-qubit_fig1_312023141
    def CCPhasePi():
        qml.CNOT(wires=[1,2])
        #T dagger
        qml.U3(0,-np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[0,2])            
        #T
        qml.U3(0,np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[1,2])
        #T dagger
        qml.U3(0,-np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[0,2])    
        #two T gates
        qml.U3(0,np.pi/4,0,wires=1)
        qml.U3(0,np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[0,1])
        #two T gates
        qml.U3(0,np.pi/4,0,wires=0)
        qml.U3(0,-np.pi/4,0,wires=1)
        
        qml.CNOT(wires=[0,1])
        
    #Note that CCX is a Toffoli gate.
    #The decomposition below of it comes from the Wikipedia page on Toffoli gates:
    #https://en.wikipedia.org/wiki/Toffoli_gate#/media/File:Qcircuit_ToffolifromCNOT.svg    
    def CCX():
        #Hadamard gate
        qml.U3(np.pi/2,0,np.pi,wires=2)
        
        qml.CNOT(wires=[1,2])
        
        #T dagger
        qml.U3(0,-np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[0,2])
        
        #T gate
        qml.U3(0,np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[1,2])
        
        #T dagger
        qml.U3(0,-np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[0,2])
        
        #two T gates
        qml.U3(0,np.pi/4,0,wires=1)
        qml.U3(0,np.pi/4,0,wires=2)
        
        qml.CNOT(wires=[0,1]) 
        
        #Hadamard gate
        qml.U3(np.pi/2,0,np.pi,wires=2)
        
        #T gate
        qml.U3(0,np.pi/4,0,wires=0)
        
        #T dagger 
        qml.U3(0,-np.pi/4,0,wires=1)
        
        qml.CNOT(wires=[0,1]) 
        
    #Pauli X gate
    qml.U3(np.pi,0,0,wires=0)

    CCPhasePi()    
    CCX()
    CCPhasePi()
    CCX()
    
    #Pauli X gate
    qml.U3(np.pi,0,0,wires=0)
    
    #Hadamard gate plus a phase of Pi
    qml.U3(3*np.pi/2,np.pi,0,wires=2)
    
# These functions are responsible for testing the solution.

def run(input: str) -> str:
    matrix = qml.matrix(circuit)().real

    with qml.tape.QuantumTape() as tape:
        circuit()

    names = [op.name for op in tape.operations]
    return json.dumps({"matrix": matrix.tolist(), "gates": names})

def check(user_output: str, expected_output: str) -> str:
    parsed_output = json.loads(user_output)
    matrix_user = np.array(parsed_output["matrix"])
    gates = parsed_output["gates"]

    solution = (
        1
        / np.sqrt(2)
        * np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, -1, 0, 0, 0, 0],
                [0, 0, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, -1],
            ]
        )
    )

    assert np.allclose(matrix_user, solution)
    assert len(set(gates)) == 2 and "U3" in gates and "CNOT" in gates


test_cases = [['No input', 'No output']]

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