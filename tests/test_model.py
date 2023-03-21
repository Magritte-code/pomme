from p3droslo.model import TensorModel


def test_constructor():
    
    
    model = TensorModel(
        var_keys  = ['CO'],
        box_sizes = (1.0, 1.0, 1.0),
        box_shape = (100, 100, 100),
    )