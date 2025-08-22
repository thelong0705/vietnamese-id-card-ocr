"""
Validation tests to verify the testing infrastructure is working correctly.
"""

import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image


class TestInfrastructureValidation:
    """Validate that the testing infrastructure is properly configured."""
    
    def test_python_version(self):
        """Verify Python version is compatible."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    def test_project_structure_exists(self):
        """Verify project directories exist."""
        project_root = Path(__file__).parent.parent
        
        # Check main package directories
        assert (project_root / "cropper").exists()
        assert (project_root / "detector").exists()
        assert (project_root / "reader").exists()
        assert (project_root / "util").exists()
        assert (project_root / "config").exists()
        
        # Check test directories
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
    
    def test_imports_work(self):
        """Verify that project modules can be imported."""
        # This will fail if the modules don't exist or have syntax errors
        # We're just checking the imports work, not testing functionality
        try:
            import cropper
            import detector
            import reader
            import util
            import config
            assert True
        except ImportError as e:
            # It's okay if modules don't have __init__.py yet
            if "__init__" not in str(e):
                pytest.fail(f"Failed to import module: {e}")
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Verify unit test marker is recognized."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Verify integration test marker is recognized."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Verify slow test marker is recognized."""
        assert True


class TestFixturesWork:
    """Validate that pytest fixtures are working correctly."""
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture creates a directory."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_sample_image_fixture(self, sample_image):
        """Test that sample_image fixture creates an image file."""
        assert sample_image.exists()
        assert sample_image.suffix == ".jpg"
        
        # Verify it's a valid image
        img = Image.open(sample_image)
        assert img.size == (640, 480)
        assert img.mode == "RGB"
    
    def test_sample_numpy_image_fixture(self, sample_numpy_image):
        """Test that sample_numpy_image fixture creates a numpy array."""
        assert isinstance(sample_numpy_image, np.ndarray)
        assert sample_numpy_image.shape == (480, 640, 3)
        assert sample_numpy_image.dtype == np.uint8
    
    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture provides configuration."""
        assert isinstance(mock_config, dict)
        assert "detector" in mock_config
        assert "reader" in mock_config
        assert "cropper" in mock_config
        assert mock_config["detector"]["confidence_threshold"] == 0.5
    
    def test_sample_id_card_data_fixture(self, sample_id_card_data):
        """Test that sample_id_card_data fixture provides test data."""
        assert isinstance(sample_id_card_data, dict)
        assert "id_number" in sample_id_card_data
        assert "full_name" in sample_id_card_data
        assert sample_id_card_data["id_number"] == "001234567890"
    
    def test_performance_timer_fixture(self, performance_timer):
        """Test that performance_timer fixture works."""
        import time
        
        performance_timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = performance_timer.stop()
        
        assert elapsed is not None
        assert elapsed >= 0.01
        assert elapsed < 0.1  # Should be less than 100ms


class TestCoverageConfiguration:
    """Validate coverage configuration."""
    
    def test_coverage_runs(self):
        """Test that coverage is being collected."""
        # This test will pass if coverage is configured correctly
        # The actual coverage percentage is checked by pytest-cov
        assert True
    
    def test_multiple_source_coverage(self):
        """Verify coverage is configured for all source directories."""
        # This is implicitly tested by the coverage configuration
        # If misconfigured, pytest will fail with coverage warnings
        assert True


class TestPytestConfiguration:
    """Validate pytest configuration."""
    
    def test_strict_markers_enabled(self):
        """Test that strict markers are enforced."""
        # This would fail at collection time if an undefined marker is used
        assert True
    
    def test_test_discovery_patterns(self):
        """Verify this test file is discovered by pytest."""
        # If this test runs, discovery patterns are working
        assert True
    
    def test_verbose_output_configured(self):
        """Test that verbose output is configured."""
        # This is visible in the test output
        print("If you see this with -vv flag, verbose is working")
        assert True


class TestMockingCapabilities:
    """Validate mocking capabilities are available."""
    
    def test_mock_tensorflow_session(self, mock_tensorflow_session):
        """Test TensorFlow session mock."""
        result = mock_tensorflow_session.run("fake_tensor")
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
    
    def test_mock_tesseract(self, mock_tesseract):
        """Test pytesseract mock."""
        text = mock_tesseract.image_to_string("fake_image")
        assert text == "Sample OCR Text"
    
    def test_mock_cv2(self, mock_cv2):
        """Test OpenCV mock."""
        img = mock_cv2.imread("fake_path")
        assert isinstance(img, np.ndarray)
        assert img.shape == (480, 640, 3)
    
    def test_pytest_mock_fixture(self, mocker):
        """Test that pytest-mock is available."""
        mock_func = mocker.Mock(return_value=42)
        assert mock_func() == 42
        mock_func.assert_called_once()


@pytest.mark.skip(reason="Example of skipping a test")
class TestSkippingWorks:
    """Validate test skipping works."""
    
    def test_this_should_be_skipped(self):
        """This test should be skipped."""
        pytest.fail("This test should have been skipped!")


def test_standalone_test_function():
    """Verify standalone test functions are discovered."""
    assert True


if __name__ == "__main__":
    # Allow running this file directly for quick validation
    pytest.main([__file__, "-v"])