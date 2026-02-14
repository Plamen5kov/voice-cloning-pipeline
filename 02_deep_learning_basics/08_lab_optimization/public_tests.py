import numpy as np

def update_parameters_with_gd_test(target):
    np.random.seed(1)
    learning_rate = 0.01
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)
    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    parameters = target(parameters, grads, learning_rate)
    
    assert type(parameters) == dict, "Wrong type for parameters. Expected a dictionary"
    assert parameters["W1"].shape == W1.shape, f"Wrong shape for W1. {parameters['W1'].shape} != {W1.shape}"
    assert parameters["b1"].shape == b1.shape, f"Wrong shape for b1. {parameters['b1'].shape} != {b1.shape}"  
    assert parameters["W2"].shape == W2.shape, f"Wrong shape for W2. {parameters['W2'].shape} != {W2.shape}"
    assert parameters["b2"].shape == b2.shape, f"Wrong shape for b2. {parameters['b2'].shape} != {b2.shape}" 
    
    assert np.allclose(parameters["W1"], np.array([[ 1.63535156, -0.62320365, -0.53718766],
       [-1.07799357,  0.85639907, -2.29470142]])), "Wrong values for W1"
    assert np.allclose(parameters["b1"], np.array([[ 1.74604067],
       [-0.75184921]])), "Wrong values for b1"
    assert np.allclose(parameters["W2"], np.array([[ 0.32171798, -0.25467393,  1.46902454],
       [-2.05617317, -0.31554548, -0.3756023 ],
       [ 1.1404819 , -1.09976462, -0.1612551 ]])), "Wrong values for W2"
    assert np.allclose(parameters["b2"], np.array([[-0.88020257],
       [ 0.02561572],
       [ 0.57539477]])), "Wrong values for b2"
    
    print("\033[92mAll tests passed!")


def random_mini_batches_test(target):
    np.random.seed(1)
    mini_batch_size = 64
    nx = 12288
    m = 148
    X = np.array([x for x in range(nx * m)]).reshape((m, nx)).T
    Y = np.random.randn(1, m) < 0.5
    
    mini_batches = target(X, Y, mini_batch_size, 0)
    
    assert len(mini_batches) == 3, f"Wrong number of mini batches. Expected 3 but got {len(mini_batches)}"
    
    for i in range(3):
        assert type(mini_batches[i]) == tuple, f"Wrong type for mini_batches[{i}]. Expected a tuple"
        
    assert mini_batches[0][0].shape == (12288, 64), f"Wrong shape for mini_batches[0][0]. {mini_batches[0][0].shape} != (12288, 64)"
    assert mini_batches[1][0].shape == (12288, 64), f"Wrong shape for mini_batches[1][0]. {mini_batches[1][0].shape} != (12288, 64)"
    assert mini_batches[2][0].shape == (12288, 20), f"Wrong shape for mini_batches[2][0]. {mini_batches[2][0].shape} != (12288, 20)"
    assert mini_batches[0][1].shape == (1, 64), f"Wrong shape for mini_batches[0][1]. {mini_batches[0][1].shape} != (1, 64)"
    assert mini_batches[1][1].shape == (1, 64), f"Wrong shape for mini_batches[1][1]. {mini_batches[1][1].shape} != (1, 64)"
    assert mini_batches[2][1].shape == (1, 20), f"Wrong shape for mini_batches[2][1]. {mini_batches[2][1].shape} != (1, 20)"
    
    assert np.allclose(mini_batches[0][0][0][0:3], [294912,  86016, 454656]), "Wrong values in mini_batches[0][0][0][0:3]"
    assert np.allclose(mini_batches[1][0][-1][0:3], [172031, 1290239, 1474559]), "Wrong values in mini_batches[1][0][-1][0:3]"
    assert np.allclose(mini_batches[2][0][-1][0:3], [1425407, 1769471, 897023]), "Wrong values in mini_batches[2][0][-1][0:3]"
    
    print("\033[92mAll tests passed!")


def initialize_velocity_test(target):
    np.random.seed(1)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    v = target(parameters)
    
    assert type(v) == dict, "Wrong type for v. Expected a dictionary"
    assert len(v) == 4, f"Wrong number of elements in v dictionary. Expected 4 but got {len(v)}"
    assert "dW1" in v, "Missing dW1 in returned dictionary v"
    assert "db1" in v, "Missing db1 in returned dictionary v"
    assert "dW2" in v, "Missing dW2 in returned dictionary v"
    assert "db2" in v, "Missing db2 in returned dictionary v"
    
    assert v["dW1"].shape == W1.shape, f"Wrong shape for v['dW1']. {v['dW1'].shape} != {W1.shape}"
    assert v["db1"].shape == b1.shape, f"Wrong shape for v['db1']. {v['db1'].shape} != {b1.shape}"
    assert v["dW2"].shape == W2.shape, f"Wrong shape for v['dW2']. {v['dW2'].shape} != {W2.shape}"
    assert v["db2"].shape == b2.shape, f"Wrong shape for v['db2']. {v['db2'].shape} != {b2.shape}"
    
    assert np.allclose(v["dW1"], np.zeros((2, 3))), "Wrong values for v['dW1']"
    assert np.allclose(v["db1"], np.zeros((2, 1))), "Wrong values for v['db1']"
    assert np.allclose(v["dW2"], np.zeros((3, 3))), "Wrong values for v['dW2']"
    assert np.allclose(v["db2"], np.zeros((3, 1))), "Wrong values for v['db2']"
    
    print("\033[92mAll tests passed!")


def update_parameters_with_momentum_test(target):
    np.random.seed(1)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    v = {'dW1': np.array([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
        [ 0.]]), 'db2': np.array([[ 0.],
        [ 0.],
        [ 0.]])}
    
    parameters, v = target(parameters, grads, v, beta=0.9, learning_rate=0.01)
    
    assert type(parameters) == dict, "Wrong type for parameters. Expected a dictionary"
    assert type(v) == dict, "Wrong type for v. Expected a dictionary"
    
    assert np.allclose(parameters["W1"], np.array([[ 1.62544598, -0.61290114, -0.52907334],
       [-1.07347112,  0.86450677, -2.30085497]])), "Wrong values for parameters['W1']"
    assert np.allclose(parameters["b1"], np.array([[ 1.74493465],
       [-0.76027113]])), "Wrong values for parameters['b1']"
    assert np.allclose(parameters["W2"], np.array([[ 0.31930698, -0.24990073,  1.4627996 ],
       [-2.05974396, -0.32173003, -0.38320915],
       [ 1.13444069, -1.0998786 , -0.1713109 ]])), "Wrong values for parameters['W2']"
    assert np.allclose(parameters["b2"], np.array([[-0.87809283],
       [ 0.04055394],
       [ 0.58207317]])), "Wrong values for parameters['b2']"
    
    assert np.allclose(v["dW1"], np.array([[-0.11006192,  0.11447237,  0.09015907],
       [ 0.05024943,  0.09008559, -0.06837279]])), "Wrong values for v['dW1']"
    assert np.allclose(v["db1"], np.array([[-0.01228902],
       [-0.09357694]])), "Wrong values for v['db1']"
    assert np.allclose(v["dW2"], np.array([[-0.02678881,  0.05303555, -0.06916608],
       [-0.03967535, -0.06871727, -0.08452056],
       [-0.06712461, -0.00126646, -0.11173103]])), "Wrong values for v['dW2']"
    assert np.allclose(v["db2"], np.array([[ 0.02344157],
       [ 0.16598022],
       [ 0.07420442]])), "Wrong values for v['db2']"
    
    print("\033[92mAll tests passed!")


def initialize_adam_test(target):
    np.random.seed(1)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    v, s = target(parameters)
    
    assert type(v) == dict, "Wrong type for v. Expected a dictionary"
    assert type(s) == dict, "Wrong type for s. Expected a dictionary"
    assert len(v) == 4, f"Wrong number of elements in v dictionary. Expected 4 but got {len(v)}"
    assert len(s) == 4, f"Wrong number of elements in s dictionary. Expected 4 but got {len(s)}"
    
    assert v["dW1"].shape == W1.shape, f"Wrong shape for v['dW1']. {v['dW1'].shape} != {W1.shape}"
    assert v["db1"].shape == b1.shape, f"Wrong shape for v['db1']. {v['db1'].shape} != {b1.shape}"
    assert v["dW2"].shape == W2.shape, f"Wrong shape for v['dW2']. {v['dW2'].shape} != {W2.shape}"
    assert v["db2"].shape == b2.shape, f"Wrong shape for v['db2']. {v['db2'].shape} != {b2.shape}"
    
    assert s["dW1"].shape == W1.shape, f"Wrong shape for s['dW1']. {s['dW1'].shape} != {W1.shape}"
    assert s["db1"].shape == b1.shape, f"Wrong shape for s['db1']. {s['db1'].shape} != {b1.shape}"
    assert s["dW2"].shape == W2.shape, f"Wrong shape for s['dW2']. {s['dW2'].shape} != {W2.shape}"
    assert s["db2"].shape == b2.shape, f"Wrong shape for s['db2']. {s['db2'].shape} != {b2.shape}"
    
    assert np.allclose(v["dW1"], np.zeros((2, 3))), "Wrong values for v['dW1']"
    assert np.allclose(v["db1"], np.zeros((2, 1))), "Wrong values for v['db1']"
    assert np.allclose(v["dW2"], np.zeros((3, 3))), "Wrong values for v['dW2']"
    assert np.allclose(v["db2"], np.zeros((3, 1))), "Wrong values for v['db2']"
    
    assert np.allclose(s["dW1"], np.zeros((2, 3))), "Wrong values for s['dW1']"
    assert np.allclose(s["db1"], np.zeros((2, 1))), "Wrong values for s['db1']"
    assert np.allclose(s["dW2"], np.zeros((3, 3))), "Wrong values for s['dW2']"
    assert np.allclose(s["db2"], np.zeros((3, 1))), "Wrong values for s['db2']"
    
    print("\033[92mAll tests passed!")


def update_parameters_with_adam_test(target):
    np.random.seed(1)
    v, s = ({'dW1': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
         [ 0.]]), 'db2': np.array([[ 0.],
         [ 0.],
         [ 0.]])}, {'dW1': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
         [ 0.]]), 'db2': np.array([[ 0.],
         [ 0.],
         [ 0.]])})
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    t = 2
    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    parameters, v, s, vc, sc = target(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
    
    assert type(parameters) == dict, "Wrong type for parameters. Expected a dictionary"
    assert type(v) == dict, "Wrong type for v. Expected a dictionary"
    assert type(s) == dict, "Wrong type for s. Expected a dictionary"
    
    assert np.allclose(parameters["W1"], np.array([[ 1.63178673, -0.61919778, -0.53561312],
       [-1.08040999,  0.85796626, -2.29409733]])), "Wrong values for parameters['W1']"
    assert np.allclose(parameters["b1"], np.array([[ 1.75225313],
       [-0.75376553]])), "Wrong values for parameters['b1']"
    assert np.allclose(parameters["W2"], np.array([[ 0.32648046, -0.25681174,  1.46954931],
       [-2.05269934, -0.31497584, -0.37661299],
       [ 1.14121081, -1.09244991, -0.16498684]])), "Wrong values for parameters['W2']"
    assert np.allclose(parameters["b2"], np.array([[-0.88529979],
       [ 0.03477238],
       [ 0.57537385]])), "Wrong values for parameters['b2']"
    
    print("\033[92mAll tests passed!")


def update_lr_test(target):
    learning_rate = 0.5
    epoch_num = 2
    decay_rate = 1
    learning_rate_2 = target(learning_rate, epoch_num, decay_rate)
    
    assert learning_rate_2 == 0.16666666666666666, f"Wrong value for learning_rate_2. Expected 0.16666666666666666 but got {learning_rate_2}"
    
    print("\033[92mAll tests passed!")


def schedule_lr_decay_test(target):
    learning_rate = 0.5
    epoch_num_1 = 10
    epoch_num_2 = 100
    decay_rate = 0.3
    time_interval = 100
    learning_rate_1 = target(learning_rate, epoch_num_1, decay_rate, time_interval)
    learning_rate_2 = target(learning_rate, epoch_num_2, decay_rate, time_interval)
    
    assert np.isclose(learning_rate_1, 0.5), f"Wrong value for learning_rate_1. Expected 0.5 but got {learning_rate_1}"
    assert np.isclose(learning_rate_2, 0.3846153846153846), f"Wrong value for learning_rate_2. Expected 0.3846153846153846 but got {learning_rate_2}"
    
    print("\033[92mAll tests passed!")
