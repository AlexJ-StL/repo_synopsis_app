# test_traverse_directory.py

import os
import tempfile
import shutil
import pytest
from unittest import mock

# Absolute import as per instructions
from streamlit_app.streamlit_app import traverse_directory

@pytest.mark.usefixtures("tmp_path")
class TestTraverseDirectory:
    # ------------------- Happy Path Tests -------------------

    @pytest.mark.happy_path
    def test_traverse_single_file(self, tmp_path):
        """Test traversing a directory with a single file returns the correct path."""
        file_path = tmp_path / "file1.txt"
        file_path.write_text("hello")
        result = traverse_directory(str(tmp_path))
        assert len(result) == 1
        assert os.path.normpath(str(file_path)) in result

    @pytest.mark.happy_path
    def test_traverse_multiple_files_and_subdirs(self, tmp_path):
        """Test traversing a directory with multiple files and subdirectories."""
        # Create files and subdirectories
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file3 = subdir / "file3.txt"
        file1.write_text("a")
        file2.write_text("b")
        file3.write_text("c")
        result = traverse_directory(str(tmp_path))
        expected = {
            os.path.normpath(str(file1)),
            os.path.normpath(str(file2)),
            os.path.normpath(str(file3)),
        }
        assert set(result) == expected

    @pytest.mark.happy_path
    def test_traverse_empty_directory(self, tmp_path):
        """Test traversing an empty directory returns an empty list."""
        result = traverse_directory(str(tmp_path))
        assert result == []

    @pytest.mark.happy_path
    def test_traverse_nested_directories(self, tmp_path):
        """Test traversing deeply nested directories with files."""
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        file1 = nested / "deepfile.txt"
        file1.write_text("deep")
        result = traverse_directory(str(tmp_path))
        assert os.path.normpath(str(file1)) in result
        assert len(result) == 1

    # ------------------- Edge Case Tests -------------------

    @pytest.mark.edge_case
    def test_traverse_nonexistent_directory(self):
        """Test traversing a non-existent directory returns empty list and warns."""
        non_existent = "/tmp/this/path/should/not/exist"
        with mock.patch("streamlit_app.streamlit_app.st.warning") as mock_warning:
            result = traverse_directory(non_existent)
            assert result == []
            assert mock_warning.called
            assert "Could not fully traverse directory" in mock_warning.call_args[0][0]

    @pytest.mark.edge_case
    def test_traverse_permission_error(self, tmp_path):
        """Test traversing a directory where a PermissionError is raised."""
        # Patch os.walk to raise PermissionError
        with mock.patch("os.walk", side_effect=PermissionError("No permission")), \
             mock.patch("streamlit_app.streamlit_app.st.warning") as mock_warning:
            result = traverse_directory(str(tmp_path))
            assert result == []
            assert mock_warning.called
            assert "Could not fully traverse directory" in mock_warning.call_args[0][0]

    @pytest.mark.edge_case
    def test_traverse_oserror(self, tmp_path):
        """Test traversing a directory where an OSError is raised."""
        with mock.patch("os.walk", side_effect=OSError("OS error")), \
             mock.patch("streamlit_app.streamlit_app.st.warning") as mock_warning:
            result = traverse_directory(str(tmp_path))
            assert result == []
            assert mock_warning.called
            assert "Could not fully traverse directory" in mock_warning.call_args[0][0]

    @pytest.mark.edge_case
    def test_traverse_directory_with_special_characters(self, tmp_path):
        """Test traversing a directory with files having special characters in names."""
        special_file = tmp_path / "spécial_文件.txt"
        special_file.write_text("data")
        result = traverse_directory(str(tmp_path))
        assert os.path.normpath(str(special_file)) in result
        assert len(result) == 1

    @pytest.mark.edge_case
    def test_traverse_directory_with_symlink(self, tmp_path):
        """Test traversing a directory containing a symlink to a file."""
        target = tmp_path / "target.txt"
        target.write_text("target")
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(target)
        result = traverse_directory(str(tmp_path))
        # Both the real file and the symlink should be listed
        assert os.path.normpath(str(target)) in result
        assert os.path.normpath(str(symlink)) in result
        assert len(result) == 2

    @pytest.mark.edge_case
    def test_traverse_directory_with_broken_symlink(self, tmp_path):
        """Test traversing a directory containing a broken symlink."""
        broken_link = tmp_path / "broken_link.txt"
        # Create a symlink to a non-existent file
        broken_link.symlink_to(tmp_path / "does_not_exist.txt")
        result = traverse_directory(str(tmp_path))
        # The broken symlink should still be listed as a file
        assert os.path.normpath(str(broken_link)) in result
        assert len(result) == 1