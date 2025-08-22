"""
Basic validation tests that don't require numpy/pillow.
"""

import sys
from pathlib import Path
import pytest


class TestBasicInfrastructure:
    """Basic infrastructure validation tests."""
    
    def test_python_version(self):
        """Verify Python version is compatible."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    def test_project_directories_exist(self):
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
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test unit marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test slow marker works."""
        assert True
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test temp_dir fixture creates a directory."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
    
    def test_mock_config_fixture(self, mock_config):
        """Test mock_config fixture provides configuration."""
        assert isinstance(mock_config, dict)
        assert "detector" in mock_config
        assert "reader" in mock_config


def test_pytest_runs():
    """Simple test to verify pytest runs."""
    assert 1 + 1 == 2