import pennylane as qml
from pennylane import numpy as np
from pennylane.drawer import draw_mpl
import inspect
import matplotlib.pyplot as plt

# Define device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)
n_layers = 2



@qml.qnode(dev, interface="torch")
def qnode(inputs,**weights_kwargs):
    for n in range(n_layers):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.Rot(*weights_kwargs[f"rot_layer_{n}"][i], wires=i)
        for i in range(n_qubits):
            qml.CRX(weights_kwargs[f"crx_layer_{n}"][i][0], wires=[i, (i + 1) % n_qubits])
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]



# @qml.qnode(dev, interface="torch")
# def qnode(inputs,**weights_kwargs):
#     qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
#     for n in range(n_layers):
        
#         for i in range(n_qubits):
#             qml.Rot(*weights_kwargs[f"rot_layer_{n}"][i], wires=i)
#         for i in range(n_qubits):
#             qml.CRX(weights_kwargs[f"crx_layer_{n}"][i][0], wires=[i, (i + 1) % n_qubits])
#     # Measurement
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def modular_w(n_qubits,n_layers):
    shapes = {}
    for n in range(n_layers):
        shapes[f"rot_layer_{n}"] = (n_qubits, 3) 
        shapes[f"crx_layer_{n}"] = (n_qubits, 1) 
    return shapes

# Call the function to have the weights for the model
weight_shapes = modular_w(n_qubits, n_layers)

def print_weight_shapes(shapes):
    print("Parameter shapes:")
    for k, v in shapes.items():
        print(f"{k:12s}: {v}")

print_weight_shapes(weight_shapes)
# style = "pennylane"
style = "black_white"
dummy_input = np.zeros(4, dtype=np.float32)
# dummy_input = np.zeros(16, dtype=np.float32)

# Create dummy weights to match weight_shapes
dummy_weights_dict = {
    name: np.zeros(shape, dtype=np.float32)
    for name, shape in weight_shapes.items()
}

qml.drawer.use_style(style)
fig, ax = draw_mpl(qnode, show_all_wires=True, fontsize=20)(dummy_input, **dummy_weights_dict)
fig.set_size_inches(12, 5)  # Wide layout
# fig.suptitle("Quantum Convolutional Neural Network Circuit", fontsize=16)
plt.tight_layout()
# plt.savefig("blk_AmplEnc-qmlp.png",dpi=600)
plt.savefig("blk_AngleEnc-qmlp.png",dpi=600)
# plt.savefig("AmplEnc-qmlp.png",dpi=600)
# plt.savefig("AngleEnc-qmlp.png",dpi=600)
# plt.show()