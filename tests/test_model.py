import torch

from pomme.model import TensorModel


def test_constructor_1D():
    model = TensorModel(
        keys  = ['CO'],
        sizes = (1.0,),
        shape = (100,),
    )


def test_constructor_2D():
    model = TensorModel(
        keys  = ['CO'],
        sizes = (1.0, 1.0),
        shape = (100, 100),
    )


def test_constructor_3D():
    model = TensorModel(
        keys  = ['CO'],
        sizes = (1.0, 1.0, 1.0),
        shape = (100, 100, 100),
    )
    

def test_is_field_1D():
    # Create a 3D model
    model = TensorModel(
        keys  = ['CO'],
        sizes = (1.0),
        shape = (100),
    )
    # Create a non-field variable
    model['non_field_var'] = torch.Tensor([1.23])
    # Check is_field
    assert model.is_field('CO')            == True
    assert model.is_field('non_field_var') == False
    

def test_is_field_3D():
    # Create a 3D model
    model = TensorModel(
        keys  = ['CO'],
        sizes = (1.0, 1.0, 1.0),
        shape = (100, 100, 100),
    )
    # Create a non-field variable
    model['non_field_var'] = torch.Tensor([1.23])
    # Check is_field
    assert model.is_field('CO')            == True
    assert model.is_field('non_field_var') == False