# test_streamlit_app_save_synopsis.py

import os
import tempfile
import shutil
import pytest
from unittest import mock

# Import the function to test
from streamlit_app.streamlit_app import save_synopsis

# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Creates a temporary directory for file operations."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

@pytest.fixture
def valid_content():
    """Provides valid content for saving."""
    return "This is a test synopsis."

@pytest.fixture
def valid_filename():
    """Provides a valid filename."""
    return "synopsis.txt"

# --- Test Class ---

class TestSaveSynopsis:
    # ------------------- Happy Path Tests -------------------

    @pytest.mark.happy_path
    def test_save_synopsis_success(self, temp_dir, valid_content, valid_filename):
        """
        Test that save_synopsis successfully writes content to a file in a valid directory.
        """
        with mock.patch("streamlit_app.streamlit_app.st"):
            result = save_synopsis(temp_dir, valid_content, valid_filename)
        assert result is True
        file_path = os.path.join(temp_dir, valid_filename)
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == valid_content

    @pytest.mark.happy_path
    def test_save_synopsis_success_with_non_ascii_content(self, temp_dir, valid_filename):
        """
        Test that save_synopsis can handle and save non-ASCII (e.g., Unicode) content.
        """
        content = "Résumé – Привет мир – こんにちは世界"
        with mock.patch("streamlit_app.streamlit_app.st"):
            result = save_synopsis(temp_dir, content, valid_filename)
        assert result is True
        file_path = os.path.join(temp_dir, valid_filename)
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == content

    # ------------------- Edge Case Tests -------------------

    @pytest.mark.edge_case
    def test_save_synopsis_empty_content(self, temp_dir, valid_filename):
        """
        Test that save_synopsis returns False and warns when content is empty.
        """
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st:
            result = save_synopsis(temp_dir, "", valid_filename)
            assert result is False
            mock_st.warning.assert_called_once_with("No content provided to save.")

    @pytest.mark.edge_case
    def test_save_synopsis_none_content(self, temp_dir, valid_filename):
        """
        Test that save_synopsis returns False and warns when content is None.
        """
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st:
            result = save_synopsis(temp_dir, None, valid_filename)
            assert result is False
            mock_st.warning.assert_called_once_with("No content provided to save.")

    @pytest.mark.edge_case
    def test_save_synopsis_invalid_directory(self, valid_content, valid_filename):
        """
        Test that save_synopsis returns False and errors when directory does not exist.
        """
        invalid_dir = "/path/does/not/exist"
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st:
            result = save_synopsis(invalid_dir, valid_content, valid_filename)
            assert result is False
            # Should call st.error with a message about invalid directory
            assert any("Invalid directory path for saving" in str(call.args[0]) for call in mock_st.error.call_args_list)

    @pytest.mark.edge_case
    def test_save_synopsis_directory_is_file(self, temp_dir, valid_content, valid_filename):
        """
        Test that save_synopsis returns False and errors when directory_path is a file, not a directory.
        """
        file_path = os.path.join(temp_dir, "not_a_dir.txt")
        with open(file_path, "w") as f:
            f.write("I'm a file, not a directory.")
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st:
            result = save_synopsis(file_path, valid_content, valid_filename)
            assert result is False
            assert any("Invalid directory path for saving" in str(call.args[0]) for call in mock_st.error.call_args_list)

    @pytest.mark.edge_case
    def test_save_synopsis_directory_path_is_empty(self, valid_content, valid_filename):
        """
        Test that save_synopsis returns False and errors when directory_path is empty.
        """
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st:
            result = save_synopsis("", valid_content, valid_filename)
            assert result is False
            assert any("Invalid directory path for saving" in str(call.args[0]) for call in mock_st.error.call_args_list)

    @pytest.mark.edge_case
    def test_save_synopsis_directory_path_is_none(self, valid_content, valid_filename):
        """
        Test that save_synopsis returns False and errors when directory_path is None.
        """
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st:
            result = save_synopsis(None, valid_content, valid_filename)
            assert result is False
            assert any("Invalid directory path for saving" in str(call.args[0]) for call in mock_st.error.call_args_list)

    @pytest.mark.edge_case
    def test_save_synopsis_directory_path_is_not_string(self, valid_content, valid_filename):
        """
        Test that save_synopsis returns False and errors when directory_path is not a string.
        """
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st:
            result = save_synopsis(12345, valid_content, valid_filename)
            assert result is False
            assert any("Invalid directory path for saving" in str(call.args[0]) for call in mock_st.error.call_args_list)

    @pytest.mark.edge_case
    def test_save_synopsis_file_write_error(self, temp_dir, valid_content, valid_filename):
        """
        Test that save_synopsis returns False and logs error if file writing fails (e.g., permission error).
        """
        # Simulate open() raising an OSError
        with mock.patch("streamlit_app.streamlit_app.st") as mock_st, \
             mock.patch("builtins.open", side_effect=OSError("Permission denied")), \
             mock.patch("streamlit_app.streamlit_app.log_event") as mock_log_event:
            result = save_synopsis(temp_dir, valid_content, valid_filename)
            assert result is False
            assert any("Error saving synopsis file" in str(call.args[0]) for call in mock_st.error.call_args_list)
            mock_log_event.assert_called_once()