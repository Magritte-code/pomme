from p3droslo.model import TensorModel


def test_constructor():
    
    
    model = TensorModel(
        keys  = ['CO'],
        sizes = (1.0, 1.0, 1.0),
        shape = (100, 100, 100),
    )