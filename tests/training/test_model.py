import pytest
import torch

from src.training.model import SimpleCNN


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_model_returns_one_logit_per_image(batch_size):
    model = SimpleCNN().eval()
    with torch.no_grad():
        output = model(torch.zeros(batch_size, 3, 64, 64))
    assert output.shape == (batch_size, 1)


def test_model_rejects_images_of_the_wrong_size():
    with pytest.raises(RuntimeError):
        SimpleCNN()(torch.zeros(1, 3, 32, 32))


def test_layer_names_match_saved_checkpoints():
    # Checkpoints are keyed by these names and the drift baseline reads fc1
    layers = {name.split(".")[0] for name in SimpleCNN().state_dict()}
    assert layers == {"conv1", "conv2", "fc1", "fc2"}
