#! /usr/bin/env python3
import enum
from gates import SGate, CXGate, CZGate, HGate, CompositeCliffordGate, SwapGate, PauliZProjector, XGate
from gates import TGate, CompositeGate
import random
import numpy as np
import cPSCS
#we want to generate circuit layers according to the Sycamore network graph
#we have 54 qubits arranged in a 6x9 grid
#with the following interactions
#directly above, above and right, directly below, below and right
#we have the following types of interaction layers
# A: even rows connected to the qubit directly above 
# B: odd rows connected above and right
# C: even rows connected directly below
# D: odd rows connected below and right
#equivalently
# A: even rows connected to the qubit directly above 
# B: even rows connected below and left
# C: even rows connected directly below
# D: even rows connected above and left

#we label a qubit at coordinates (i,j) with qubit number i + j*width where i is the width coordinate and j the height

class InteractionLayerType(enum.Enum):
    A = enum.auto()
    B = enum.auto()
    C = enum.auto()
    D = enum.auto()

def sycamore_grid_connections(width, height, layerType):
    connections = []
    if layerType == InteractionLayerType.A:
        for row in range(2, height,2):
            for col in range(width):
                connections.append((col + row*width, col+(row-1)*width))
                
    if layerType == InteractionLayerType.B:
        for row in range(0, height-1,2):
            for col in range(1,width):
                connections.append((col + row*width, col-1 + (row+1)*width))
                
    if layerType == InteractionLayerType.C:
        for row in range(0, height-1, 2):
            for col in range(width):
                connections.append((col + row*width, col + (row+1)*width))
                
    if layerType == InteractionLayerType.D:
        for row in range(2, height, 2):
            for col in range(1,width):
                connections.append((col + row*width, col-1 + (row-1)*width))

    return connections


def qubit_num_to_coords(n,width,height):
    return n % width, n // width

def qubit_num_to_x(n,width,height):
    return n % width

def qubit_num_to_y(n,width,height):
    return n // width

def qubit_num_to_physical_x(n,width,height):
    return (n % width) + ((n // width) % 2)*0.5

def qubit_num_to_physical_y(n,width,height):
    return height - (n // width)



def plot_sycamore_grid_connections(width, height):
    from matplotlib import pyplot as plt
    import numpy as np

    qubits = np.array(range(54), dtype=int)

    width = 6
    height = 9

    plt.scatter(qubit_num_to_physical_x(qubits, width,height), qubit_num_to_physical_y(qubits, width,height), s=plt.rcParams['lines.markersize'] ** 4, color="0.5")

    a_connections = np.array(sycamore_grid_connections(width, height, InteractionLayerType.A), dtype=int).T
    a_connections_x = qubit_num_to_physical_x(a_connections, width, height)
    a_connections_y = qubit_num_to_physical_y(a_connections, width, height)

    plt.plot(a_connections_x, a_connections_y, linewidth=6, markersize=0, color="g", label="A")

    b_connections = np.array(sycamore_grid_connections(width, height, InteractionLayerType.B), dtype=int).T
    b_connections_x = qubit_num_to_physical_x(b_connections, width, height)
    b_connections_y = qubit_num_to_physical_y(b_connections, width, height)

    plt.plot(b_connections_x, b_connections_y, linewidth=6, markersize=0, color="c", label="B")

    c_connections = np.array(sycamore_grid_connections(width, height, InteractionLayerType.C), dtype=int).T
    c_connections_x = qubit_num_to_physical_x(c_connections, width, height)
    c_connections_y = qubit_num_to_physical_y(c_connections, width, height)

    plt.plot(c_connections_x, c_connections_y, linewidth=6, markersize=0, color="b",label="C")

    d_connections = np.array(sycamore_grid_connections(width, height, InteractionLayerType.D), dtype=int).T
    d_connections_x = qubit_num_to_physical_x(d_connections, width, height)
    d_connections_y = qubit_num_to_physical_y(d_connections, width, height)

    plt.plot(d_connections_x, d_connections_y, linewidth=6, markersize=0, color="m", label="D")
    
    
    
    plt.show()
    

def random_sycamore_single_qubit_layer(width,height,rng):
    circ = CompositeGate()
    gates = 0
    for i in range(width*height):
        if i != 3:
            gates += 1
            squared_gate = rng.choice(["X", "Y", "W"])
            if squared_gate == "X": # our gate is sqrt(X)
                circ.gates.append(HGate(i))
                circ.gates.append(SGate(i))
                circ.gates.append(HGate(i))
            if squared_gate == "Y": # our gate is sqrt(Y)
                # sqrt(Y) = S sqrt(X) S^dagger
                circ.gates.append(SGate(i))
                circ.gates.append(HGate(i))
                circ.gates.append(SGate(i))
                circ.gates.append(HGate(i))
                circ.gates.append(SGate(i))
                circ.gates.append(SGate(i))
                circ.gates.append(SGate(i))
            if squared_gate == "W": # our gate is sqrt(W)
                # sqrt(W) = T H S H T^dagger = T H S H S T 
                circ.gates.append(TGate(i))
                circ.gates.append(HGate(i))
                circ.gates.append(SGate(i))
                circ.gates.append(HGate(i))
                circ.gates.append(SGate(i))
                circ.gates.append(TGate(i))
    
    return circ, gates

def sycamore_double_qubit_layer(width,height, layerType):
    circ = CompositeGate()

    gates = 0
    for (a,b) in sycamore_grid_connections(width, height, layerType):
        if a != 3 and b != 3:
            circ.gates.append(SGate(a))
            circ.gates.append(SGate(b))
            circ.gates.append(HGate(a))
            circ.gates.append(CXGate(a,b))
            circ.gates.append(CXGate(b,a))
            circ.gates.append(HGate(b))
            
            circ.gates.append(TGate(a))
            circ.gates.append(TGate(b))
            circ.gates.append(CXGate(a,b))
            circ.gates.append(TGate(b))
            circ.gates.append(CXGate(a,b))
            gates += 1
    return circ, gates

        
    

def sycamore_circuit(width, height, cycles, rng):
    circ = CompositeGate()
    entangling_pattern = [InteractionLayerType.A,
                          InteractionLayerType.B,
                          InteractionLayerType.C,
                          InteractionLayerType.D,
                          InteractionLayerType.C,
                          InteractionLayerType.D,
                          InteractionLayerType.A,
                          InteractionLayerType.B]
    single_qubit_gate_count = 0
    entangling_gate_count = 0
    
    for i in range(cycles):
        new_section, count = random_sycamore_single_qubit_layer(width, height,rng)
        single_qubit_gate_count += count
        circ | new_section

        new_section, count = sycamore_double_qubit_layer(width, height, entangling_pattern[ i % 8])
        entangling_gate_count += count
        circ | new_section
        
    new_section, count = random_sycamore_single_qubit_layer(width, height, rng)
    single_qubit_gate_count += count
    circ | new_section

    return circ, single_qubit_gate_count, entangling_gate_count
          
        
            
    
def sycamore():
    height = 9
    width = 6
    plot_sycamore_grid_connections(6,9)
    circ, single_qubit_gate_count, entangling_gate_count = sycamore_circuit(width, height, 8)
    #print("Single qubit gates: ", single_qubit_gate_count)
    #print("Entangling gates: ", entangling_gate_count)

    ts = 0
    for gate in circ.gates:
        if isinstance(gate, TGate):
            ts += 1
    print("total t = ", ts)
    print("total gates = ", len(circ.gates))
    depth = len(circ.gates)
    gateArray = np.zeros(depth, dtype=np.uint8)
    controlArray = np.zeros(depth, dtype=np.uint)
    targetArray = np.zeros(depth, dtype=np.uint)
    #qkcirc = qiskit.QuantumCircuit(qubits)

    for j, gate in enumerate(circ.gates):
        if isinstance(gate, CXGate):
            gateArray[j] = 88 #X
            controlArray[j] = gate.control
            targetArray[j] = gate.target
        elif isinstance(gate, CZGate):
            gateArray[j] = 90 #Z
            controlArray[j] = gate.control
            targetArray[j] = gate.target
        elif isinstance(gate, SGate):
            gateArray[j] = 115 #s
            targetArray[j] = gate.target
        elif isinstance(gate, HGate):
            gateArray[j] = 104 #h
            targetArray[j] = gate.target
        elif isinstance(gate, TGate):
            gateArray[j] = 116 # t
            targetArray[j] = gate.target


    #there is no qubit 3
    for i in range(len(circ.gates)):
        if controlArray[i] > 3:
            controlArray[i] = controlArray[i] - 1
        if targetArray[i] > 3:
            targetArray[i] = targetArray[i] - 1
    
    qubits = 53 # 53 qubits labelled 0 to 52
    
    measured_qubits = 5 # qubits
    aArray = np.random.randint(0,2, size=qubits,dtype=np.uint8)

    out = cPSCS.compress_algorithm(qubits, measured_qubits, gateArray, controlArray, targetArray, aArray)
    if type(out) == tuple:
        d, r, t, delta_d, delta_t, delta_t_prime, final_d, final_t, log_v, pyChState, pyAGState, magic_arr = out
        print("final_t = ", final_t)
        print("r = ", r)
        print("log_v = ", log_v)
        print("d = ", d)
        print("final_d = ", final_d)
        return True
    else:
        #print(out)
        #print("inconsistent measurement")
        return False
                

if __name__ == "__main__":
    random.seed()
    consistent = 0
    inconsistent = 0
    for  i in range(10000):
        print(i)
        if sycamore():
            consistent += 1
        else:
            inconsistent += 1
    print(consistent)
    print(inconsistent)
