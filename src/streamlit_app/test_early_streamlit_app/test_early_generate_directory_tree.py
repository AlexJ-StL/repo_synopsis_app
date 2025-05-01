# test_streamlit_app_generate_directory_tree.py

import os
import tempfile
import shutil
import pytest

from streamlit_app.streamlit_app import generate_directory_tree, RepoData, FileData, generate_directory_tree

@pytest.mark.usefixtures("setup_test_dirs")
class TestGenerateDirectoryTree:
    """Unit tests for generate_directory_tree function."""

    @pytest.fixture(autouse=True)
    def setup_test_dirs(self, tmp_path):
        """
        Shared fixture to create a temporary directory for each test.
        """
        self.tmp_path = tmp_path

    # ------------------- Happy Path Tests -------------------

    @pytest.mark.happy
    def test_single_file_in_root(self):
        """
        Test that a directory with a single file in the root is represented correctly.
        """
        file_path = self.tmp_path / "file1.txt"
        file_path.write_text("hello")
        expected = "  file1.txt"
        result = generate_directory_tree(str(self.tmp_path))
        assert result.strip() == expected

    @pytest.mark.happy
    def test_nested_directories_and_files(self):
        """
        Test that nested directories and files are represented with correct indentation and order.
        """
        # Structure:
        # root/
        #   file1.txt
        #   dirA/
        #     file2.txt
        #     dirB/
        #       file3.txt
        (self.tmp_path / "file1.txt").write_text("root file")
        dirA = self.tmp_path / "dirA"
        dirA.mkdir()
        (dirA / "file2.txt").write_text("A file")
        dirB = dirA / "dirB"
        dirB.mkdir()
        (dirB / "file3.txt").write_text("B file")

        expected = "\n".join([
            "  file1.txt",
            "  dirA/",
            "    file2.txt",
            "    dirB/",
            "      file3.txt"
        ])
        result = generate_directory_tree(str(self.tmp_path))
        assert result.strip() == expected

    @pytest.mark.happy
    def test_multiple_files_and_dirs_sorted(self):
        """
        Test that files and directories are sorted alphabetically in the output.
        """
        (self.tmp_path / "b.txt").write_text("b")
        (self.tmp_path / "a.txt").write_text("a")
        dir1 = self.tmp_path / "dir1"
        dir2 = self.tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir2 / "z.txt").write_text("z")
        (dir1 / "y.txt").write_text("y")

        expected = "\n".join([
            "  a.txt",
            "  b.txt",
            "  dir1/",
            "    y.txt",
            "  dir2/",
            "    z.txt"
        ])
        result = generate_directory_tree(str(self.tmp_path))
        assert result.strip() == expected

    @pytest.mark.happy
    def test_empty_directory(self):
        """
        Test that an empty directory returns an empty string.
        """
        result = generate_directory_tree(str(self.tmp_path))
        assert result.strip() == ""

    # ------------------- Edge Case Tests -------------------

    @pytest.mark.edge
    def test_directory_with_only_subdirectories(self):
        """
        Test that a directory with only subdirectories (no files) is represented correctly.
        """
        (self.tmp_path / "dirA").mkdir()
        (self.tmp_path / "dirB").mkdir()
        expected = "\n".join([
            "  dirA/",
            "  dirB/"
        ])
        result = generate_directory_tree(str(self.tmp_path))
        # Only directories, no files inside them
        assert result.strip() == expected

    @pytest.mark.edge
    def test_directory_with_hidden_files_and_dirs(self):
        """
        Test that hidden files and directories (starting with .) are included and sorted.
        """
        (self.tmp_path / ".hiddenfile").write_text("hidden")
        (self.tmp_path / ".hiddendir").mkdir()
        (self.tmp_path / ".hiddendir" / "afile.txt").write_text("afile")
        expected = "\n".join([
            "  .hiddenfile",
            "  .hiddendir/",
            "    afile.txt"
        ])
        result = generate_directory_tree(str(self.tmp_path))
        assert result.strip() == expected

    @pytest.mark.edge
    def test_nonexistent_directory(self):
        """
        Test that a non-existent directory returns an error message.
        """
        non_existent = self.tmp_path / "does_not_exist"
        result = generate_directory_tree(str(non_existent))
        assert result.startswith("Error generating tree:")

    @pytest.mark.edge
    def test_file_instead_of_directory(self):
        """
        Test that passing a file path instead of a directory returns an empty string (since os.walk yields nothing).
        """
        file_path = self.tmp_path / "file.txt"
        file_path.write_text("data")
        result = generate_directory_tree(str(file_path))
        # os.walk on a file yields nothing, so should return empty string
        assert result.strip() == ""

    @pytest.mark.edge
    def test_directory_with_unicode_names(self):
        """
        Test that directories and files with unicode names are handled correctly.
        """
        (self.tmp_path / "文件.txt").write_text("unicode file")
        (self.tmp_path / "目录").mkdir()
        (self.tmp_path / "目录" / "файл.txt").write_text("cyrillic file")
        expected = "\n".join([
            "  文件.txt",
            "  目录/",
            "    файл.txt"
        ])
        result = generate_directory_tree(str(self.tmp_path))
        assert result.strip() == expected

    @pytest.mark.edge
    def test_directory_with_symlink(self):
        """
        Test that symlinks to files and directories are included in the output.
        """
        # Create a real file and a symlink to it
        real_file = self.tmp_path / "real.txt"
        real_file.write_text("real")
        symlink_file = self.tmp_path / "link.txt"
        symlink_file.symlink_to(real_file)
        # Create a real dir and a symlink to it
        real_dir = self.tmp_path / "real_dir"
        real_dir.mkdir()
        (real_dir / "inside.txt").write_text("inside")
        symlink_dir = self.tmp_path / "link_dir"
        symlink_dir.symlink_to(real_dir, target_is_directory=True)

        # Symlinks are treated as files or directories by os.walk depending on the system and Python version.
        # os.walk by default does not follow directory symlinks unless followlinks=True (which is not set).
        # So, link_dir will appear as a file, not as a directory.
        # The output should include both symlinks as files.
        expected_lines = [
            "  link.txt",
            "  real.txt",
            "  real_dir/",
            "    inside.txt",
            "  link_dir"
        ]
        result = generate_directory_tree(str(self.tmp_path))
        # The order may vary for symlinks, so sort both for comparison
        assert sorted(result.strip().splitlines()) == sorted(expected_lines)