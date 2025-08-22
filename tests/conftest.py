"""
Shared pytest fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock, Mock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "detector": {
            "model_path": "/path/to/model",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4
        },
        "reader": {
            "language": "vie",
            "config": "--psm 7"
        },
        "cropper": {
            "padding": 10,
            "aspect_ratio": 1.586
        },
        "debug": False,
        "output_dir": "/tmp/output"
    }


@pytest.fixture
def mock_tensorflow_session():
    """Mock TensorFlow session for testing without GPU."""
    mock_session = MagicMock()
    mock_session.run = MagicMock(return_value=[
        [[0.1, 0.2, 0.8, 0.9]],  # Mock detection boxes
        [0.95],  # Mock detection scores
        [1]  # Mock detection classes
    ])
    return mock_session


@pytest.fixture
def mock_tesseract():
    """Mock pytesseract for OCR testing."""
    mock = MagicMock()
    mock.image_to_string = MagicMock(return_value="Sample OCR Text")
    mock.image_to_data = MagicMock(return_value={
        'text': ['Sample', 'OCR', 'Text'],
        'conf': [95, 92, 98]
    })
    return mock


@pytest.fixture
def mock_cv2():
    """Mock OpenCV functions."""
    mock = MagicMock()
    mock.imread = MagicMock(return_value=None)
    mock.imwrite = MagicMock(return_value=True)
    mock.resize = MagicMock(side_effect=lambda img, size: None)
    return mock


@pytest.fixture
def sample_id_card_data() -> Dict[str, str]:
    """Sample Vietnamese ID card data for testing."""
    return {
        "id_number": "001234567890",
        "full_name": "NGUYEN VAN A",
        "date_of_birth": "01/01/1990",
        "gender": "Nam",
        "nationality": "Viet Nam",
        "place_of_origin": "Ha Noi",
        "place_of_residence": "Ho Chi Minh",
        "date_of_expiry": "01/01/2030"
    }


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_model_file(temp_dir: Path) -> Path:
    """Create a mock model file for testing."""
    model_path = temp_dir / "test_model.pb"
    model_path.write_bytes(b"Mock model data")
    return model_path


@pytest.fixture
def performance_timer():
    """Simple performance timer for test benchmarking."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            if self.start_time:
                self.elapsed = time.perf_counter() - self.start_time
                return self.elapsed
            return None
    
    return Timer()


@pytest.fixture(scope="session")
def test_resources_dir() -> Path:
    """Path to test resources directory."""
    resources = PROJECT_ROOT / "tests" / "resources"
    resources.mkdir(exist_ok=True, parents=True)
    return resources


# Additional fixtures for numpy/pillow when installed
try:
    import numpy as np
    from PIL import Image
    
    @pytest.fixture
    def sample_image(temp_dir: Path) -> Path:
        """Create a sample test image."""
        image_path = temp_dir / "test_image.jpg"
        # Create a simple RGB image
        img = Image.new('RGB', (640, 480), color='white')
        img.save(image_path)
        return image_path
    
    @pytest.fixture
    def sample_numpy_image() -> np.ndarray:
        """Create a sample numpy array representing an image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
except ImportError:
    # These fixtures won't be available if numpy/pillow aren't installed
    pass


# Markers for conditional test execution
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_tesseract: mark test as requiring Tesseract OCR"
    )
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet connection"
    )


# Hooks for test collection and execution
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests based on naming convention
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)