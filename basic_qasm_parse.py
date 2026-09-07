#/usr/bin/env python

import gates


# we don't want to parse all of qasm2.0 because
# 1. this would be difficult
# 2. the circuits we can simulate are rather simple anyway (Clifford+T(phi) gates, computational basis measurements at the end, no classical controls)

# for simplicty we also assume that we're looking for the computational basis output |0> on all measured qubits
# we can just append X gates to the end of the circuit to do other outcomes

# we assume there is exactly one qreg containing all our qubits
def read_quantum_circuit(input_file):
    qubits = None
    circuit = gates.CompositeGate()
    measured_qubits = []
    
    for lineno, line in enumerate(input_file):
        split_line = line.split() # split on arbitrary whitespace
        if len(split_line) > 0:
            first_token = split_line[0]
            if first_token == "qreg":
                if len(split_line) > 1:
                    qubits = int(string_between_delimiters(split_line[1], "[", "]"))
                else:
                    print("Error - line {}: the qreg token should be followed by another token".format(lineno))
            #pauli operators 
            if first_token == "x":
                if len(split_line) > 1:
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    circuit.gates.append(gates.cliffords.XGate(qubit))
                else:
                    print("Error - line {}: the x token should be followed by another token".format(lineno))
            if first_token == "y":
                if len(split_line) > 1:
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    circuit.gates.append(gates.cliffords.XGate(qubit))
                    circuit.gates.append(gates.cliffords.SGate(qubit))
                    circuit.gates.append(gates.cliffords.SGate(qubit))
                else:
                    print("Error - line {}: the y token should be followed by another token".format(lineno))
            if first_token == "z":
                if len(split_line) > 1:
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    circuit.gates.append(gates.cliffords.SGate(qubit))
                    circuit.gates.append(gates.cliffords.SGate(qubit))
                else:
                    print("Error - line {}: the z token should be followed by another token".format(lineno))
            #single qubit cliffords
            if first_token == "s":
                if len(split_line) > 1:
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    circuit.gates.append(gates.cliffords.SGate(qubit))
                else:
                    print("Error - line {}: the s token should be followed by another token".format(lineno))                    
            if first_token == "h":
                if len(split_line) > 1:
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    circuit.gates.append(gates.cliffords.HGate(qubit))
                else:
                    print("Error - line {}: the h token should be followed by another token".format(lineno))
            #two qubit cliffords
            if first_token == "cx":
                if len(split_line) > 2:
                    control_qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    target_qubit = int(string_between_delimiters(split_line[2], "[", "]"))
                    circuit.gates.append(gates.cliffords.CXGate(control=control_qubit, target=target_qubit))
                else:
                    print("Error - line {}: the cx token should be followed by two tokens".format(lineno))
            if first_token == "cy":
                if len(split_line) > 2:
                    control_qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    target_qubit = int(string_between_delimiters(split_line[2], "[", "]"))
                    circuit.gates.append(gates.cliffords.SGate(target_qubit))
                    circuit.gates.append(gates.cliffords.CXGate(control=control_qubit, target=target_qubit))
                    circuit.gates.append(gates.cliffords.SGate(target_qubit))
                    circuit.gates.append(gates.cliffords.SGate(target_qubit))
                    circuit.gates.append(gates.cliffords.SGate(target_qubit))
                else:
                    print("Error - line {}: the cy token should be followed by two tokens".format(lineno))
            if first_token == "cz":
                if len(split_line) > 2:
                    control_qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    target_qubit = int(string_between_delimiters(split_line[2], "[", "]"))
                    circuit.gates.append(gates.cliffords.CZGate(control=control_qubit, target=target_qubit))
                elif len(split_line) == 2 and split_line[1].count(",") == 1:
                    p1, p2 = split_line[1].split(",")
                    control_qubit = int(string_between_delimiters(p1, "[", "]"))
                    target_qubit = int(string_between_delimiters(p2, "[", "]"))
                    circuit.gates.append(gates.cliffords.CZGate(control=control_qubit, target=target_qubit))
                else:
                    print("Error - line {}: the cz token should be followed by two tokens".format(lineno))

                                 
            #non-Clifford unitaries
            if first_token == "t":
                if len(split_line) > 1:
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    circuit.gates.append(gates.TGate(qubit))                    
                else:
                    print("Error - line {}: the h token should be followed by another token".format(lineno))
            if first_token == "rz":
                if len(split_line) > 2:
                    phase = float(string_between_delimiters(split_line[1], "(", ")"))
                    qubit = int(string_between_delimiters(split_line[2], "[", "]"))
                    circuit.gates.append(gates.RZGate(qubit, phase))                    
                else:
                    print("Error - line {}: the rz token should be followed by two tokens".format(lineno))
            if line.startswith("ry("):
                print("hello: ", lineno," ",split_line)
                if len(split_line) == 2:
                    phase = float(string_between_delimiters(split_line[0], "(", ")"))
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    circuit.gates.append(gates.SGate(qubit))
                    circuit.gates.append(gates.HGate(qubit))
                    circuit.gates.append(gates.RZGate(qubit, phase))
                    circuit.gates.append(gates.HGate(qubit))
                    circuit.gates.append(gates.SGate(qubit))
                    circuit.gates.append(gates.SGate(qubit))
                    circuit.gates.append(gates.SGate(qubit))
                else:
                    print("Error - line {}: the rz token should be followed by two tokens".format(lineno))
            if first_token == "ccz":
                if len(split_line) == 4:
                    q1, q2, q3 = list(map(int, map(lambda x: string_between_delimiters(x, "[", "]"),  split_line[1:])))
                    ccz_decomp_gates = gates.CCZGate(q1,q2, q3).gates
                    circuit.gates.extend(ccz_decomp_gates)
                else:
                    print("Error - line {}: the rz token should be followed by three tokens".format(lineno))
            #measurements
            if first_token == "measure":
                if len(split_line) > 1:
                    qubit = int(string_between_delimiters(split_line[1], "[", "]"))
                    measured_qubits.append(qubit)
                else:
                    print("Error - line {}: the measure token should be followed by another token".format(lineno))
    return qubits, measured_qubits, circuit

def string_between_delimiters(string, start, end):
    #assume the input string contains at least 1 of each of the delimiters and that there is an end delimiter after the first start delimiter
    #return the string contained between the first instance of the start delimiter
    #and the first instance of the end delimiter which is after the first instance of the end delimiter

    return string.split(start, 1)[1].split(end,1)[0]
    
