import base64
import json
from io import BytesIO

from PIL import Image

from src.training.model import SimpleCNN
from src.training.wrapper import SimpleLoggingWrapper


def _base64_image():
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_predict_does_not_log_the_image(tmp_path):
    log_path = tmp_path / "inference_log.json"
    wrapper = SimpleLoggingWrapper(model=SimpleCNN(), log_path=log_path)
    wrapper.load_context(None)

    image = _base64_image()
    result = wrapper.predict(None, {"instances": [image, image]})

    assert result["num_samples"] == 2
    assert None not in result["predictions"]

    entries = json.loads(log_path.read_text())
    assert len(entries) == 2
    for entry in entries:
        # The drift check reads these keys
        assert set(entry) == {"timestamp", "image_path", "prediction", "confidence", "raw_output"}
        assert entry["image_path"] == "<image>"
    assert image not in log_path.read_text()
