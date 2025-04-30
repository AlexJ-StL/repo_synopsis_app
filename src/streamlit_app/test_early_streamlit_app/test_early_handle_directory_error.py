# test_streamlit_app_handle_directory_error.py

import pytest
import os
from unittest import mock

# Import the function under test
from streamlit_app.streamlit_app import handle_directory_error

# --- Fixtures ---

@pytest.fixture(autouse=True)
def patch_st_error():
    """Patch st.error to prevent actual Streamlit calls and to allow assertion."""
    with mock.patch("streamlit_app.streamlit_app.st.error") as mock_error:
        yield mock_error

@pytest.fixture
def valid_tmp_dir(tmp_path):
    """Provide a valid temporary directory path."""
    return str(tmp_path)

@pytest.fixture
def file_in_tmp_dir(tmp_path):
    """Provide a file path (not a directory) inside tmp_path."""
    file_path = tmp_path / "afile.txt"
    file_path.write_text("hello")
    return str(file_path)

# --- Test Class ---

class TestHandleDirectoryError:
    # --- Happy Path Tests ---

    @pytest.mark.happy
    def test_valid_directory_returns_true(self, valid_tmp_dir, patch_st_error):
        """Test that a valid, accessible directory returns True and does not call st.error."""
        result = handle_directory_error(valid_tmp_dir)
        assert result is True
        patch_st_error.assert_not_called()

    @pytest.mark.happy
    def test_valid_directory_with_files_returns_true(self, tmp_path, patch_st_error):
        """Test that a valid directory containing files returns True."""
        (tmp_path / "file1.txt").write_text("abc")
        (tmp_path / "file2.txt").write_text("def")
        result = handle_directory_error(str(tmp_path))
        assert result is True
        patch_st_error.assert_not_called()

    # --- Edge Case Tests ---

    @pytest.mark.edge
    def test_empty_string_path_returns_false(self, patch_st_error):
        """Test that an empty string as directory path returns False and calls st.error."""
        result = handle_directory_error("")
        assert result is False
        patch_st_error.assert_called_once_with("Directory path cannot be empty.")

    @pytest.mark.edge
    def test_none_path_returns_false(self, patch_st_error):
        """Test that None as directory path returns False and calls st.error."""
        # Cast None to str for the type checker
        result = handle_directory_error(cast(str, 123))
        assert result is False
        patch_st_error.assert_called_once_with("Directory path must be a string.")
        assert result is False
        patch_st_error.assert_called_once_with("Directory path must be a string.")

    @pytest.mark.edge
    def test_non_string_path_returns_false(self, patch_st_error):
        """Test that a non-string (e.g., int) as directory path returns False and calls st.error."""
        result = handle_directory_error(123)
        assert result is False
        patch_st_error.assert_called_once_with("Directory path must be a string.")

    @pytest.mark.edge
    def test_nonexistent_directory_returns_false(self, patch_st_error):
        """Test that a non-existent directory returns False and calls st.error."""
        fake_path = "/unlikely/to/exist/dir_123456"
        result = handle_directory_error(fake_path)
        assert result is False
        patch_st_error.assert_called_once_with(f"Directory does not exist: {fake_path}")

    @pytest.mark.edge
    def test_path_is_file_not_directory_returns_false(self, file_in_tmp_dir, patch_st_error):
        """Test that a path to a file (not a directory) returns False and calls st.error."""
        result = handle_directory_error(file_in_tmp_dir)
        assert result is False
        patch_st_error.assert_called_once_with(f"Path is not a directory: {file_in_tmp_dir}")

    @pytest.mark.edge
    def test_permission_error_on_listdir_returns_false(self, valid_tmp_dir, patch_st_error):
        """Test that a PermissionError during os.listdir returns False and calls st.error."""
        with mock.patch("os.listdir", side_effect=PermissionError):
            result = handle_directory_error(valid_tmp_dir)
        assert result is False
        patch_st_error.assert_called_once_with(
            f"Permission denied accessing directory: {valid_tmp_dir}"
        )

    @pytest.mark.edge
    def test_oserror_on_listdir_returns_false(self, valid_tmp_dir, patch_st_error):
        """Test that an OSError during os.listdir returns False and calls st.error."""
        with mock.patch("os.listdir", side_effect=OSError("disk error")):
            result = handle_directory_error(valid_tmp_dir)
        assert result is False
        patch_st_error.assert_called_once()
        assert "OS error accessing directory" in patch_st_error.call_args[0][0]
        assert "disk error" in patch_st_error.call_args[0][0]

    @pytest.mark.edge
    def test_unexpected_exception_on_listdir_returns_false(self, valid_tmp_dir, patch_st_error):
        """Test that an unexpected Exception during os.listdir returns False and calls st.error."""
        class CustomException(Exception):
            pass
        with mock.patch("os.listdir", side_effect=CustomException("boom!")):
            result = handle_directory_error(valid_tmp_dir)
        assert result is False
        patch_st_error.assert_called_once()
        assert "Unexpected error validating directory" in patch_st_error.call_args[0][0]
        assert "boom!" in patch_st_error.call_args[0][0]