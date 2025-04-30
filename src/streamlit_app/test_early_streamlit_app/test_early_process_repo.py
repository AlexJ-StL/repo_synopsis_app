# test_streamlit_app_process_repo.py

import os
import tempfile
import shutil
import pytest
from unittest import mock

# Import the function under test
from streamlit_app.streamlit_app import process_repo

# --- Fixtures for test setup/teardown ---

@pytest.fixture
def temp_repo_with_files():
    """
    Creates a temporary directory with a variety of files for testing.
    Yields the directory path and a dict of file paths.
    """
    temp_dir = tempfile.mkdtemp()
    files = {
        "main.py": "print('Hello, world!')\n",
        "test_utils.py": "# test file\n def test(): pass\n",
        "README.md": "# Project\nSome description.",
        "empty.txt": "",
        "large.txt": "word " * (1024 * 1024),  # ~1MB file
        "unknownfile": "data",
        "Makefile": "all:\n\techo 'build'",
    }
    file_paths = {}
    for fname, content in files.items():
        fpath = os.path.join(temp_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        file_paths[fname] = fpath
    yield temp_dir, file_paths
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_empty_dir():
    """Creates an empty temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_st_error_warning(monkeypatch):
    """Mocks st.error and st.warning to avoid Streamlit dependency."""
    monkeypatch.setattr("streamlit_app.streamlit_app.st.error", lambda msg: None)
    monkeypatch.setattr("streamlit_app.streamlit_app.st.warning", lambda msg: None)

@pytest.fixture
def mock_get_llm_response(monkeypatch):
    """Mocks get_llm_response to return predictable values."""
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.get_llm_response",
        lambda file_path, llm_provider: (f"desc:{os.path.basename(file_path)}", f"usecase:{os.path.basename(file_path)}")
    )

# --- Test Class ---

class TestProcessRepo:
    # --- Happy Path Tests ---

    @pytest.mark.happy_path
    def test_basic_python_and_markdown_files(
        self, temp_repo_with_files, mock_st_error_warning, mock_get_llm_response
    ):
        """
        Test processing a repo with .py, .md, and other files, with all options enabled.
        Ensures correct language detection, token count, description, and use case.
        """
        repo_path, file_paths = temp_repo_with_files
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"

        result = process_repo(repo_path, include_options, llm_provider)

        assert result["repo_path"] == repo_path
        assert result["error"] is None
        # Should include all files except directories
        assert len(result["files"]) == len(file_paths)
        # Check languages
        langs = set(f["language"] for f in result["files"])
        assert "Python" in langs
        assert "Markdown" in langs
        assert "Text" in langs
        assert "Other" in langs or "Unknown" in langs or "Makefile" in langs

        # Check token_count, description, use_case for .py file
        py_file = next(f for f in result["files"] if f["path"].endswith("main.py"))
        assert isinstance(py_file["token_count"], int)
        assert py_file["description"].startswith("desc:main.py")
        assert py_file["use_case"].startswith("usecase:main.py")

        # Check test file use_case
        test_file = next(f for f in result["files"] if f["path"].endswith("test_utils.py"))
        assert test_file["use_case"] == "usecase:test_utils.py"

        # Check Makefile language
        makefile = next(f for f in result["files"] if f["path"].endswith("Makefile"))
        assert makefile["language"] == "Makefile"

    @pytest.mark.happy_path
    def test_only_token_count(self, temp_repo_with_files, mock_st_error_warning):
        """
        Test with only token_count enabled, no descriptions/use_cases.
        """
        repo_path, file_paths = temp_repo_with_files
        include_options = {"token_count": True, "descriptions": False, "use_cases": False}
        llm_provider = "dummy"

        # Patch get_llm_response to ensure it's not called
        with mock.patch("streamlit_app.streamlit_app.get_llm_response") as mock_llm:
            result = process_repo(repo_path, include_options, llm_provider)
            assert all("description" not in f and "use_case" not in f for f in result["files"])
            assert any(isinstance(f.get("token_count"), int) or isinstance(f.get("token_count"), str) for f in result["files"])
            mock_llm.assert_not_called()

    @pytest.mark.happy_path
    def test_only_descriptions(self, temp_repo_with_files, mock_st_error_warning, mock_get_llm_response):
        """
        Test with only descriptions enabled.
        """
        repo_path, file_paths = temp_repo_with_files
        include_options = {"token_count": False, "descriptions": True, "use_cases": False}
        llm_provider = "dummy"

        result = process_repo(repo_path, include_options, llm_provider)
        for f in result["files"]:
            if f["language"] not in ["Unknown", "Other"]:
                assert "description" in f
            else:
                assert "description" not in f

    @pytest.mark.happy_path
    def test_only_use_cases(self, temp_repo_with_files, mock_st_error_warning, mock_get_llm_response):
        """
        Test with only use_cases enabled.
        """
        repo_path, file_paths = temp_repo_with_files
        include_options = {"token_count": False, "descriptions": False, "use_cases": True}
        llm_provider = "dummy"

        result = process_repo(repo_path, include_options, llm_provider)
        for f in result["files"]:
            if f["language"] not in ["Unknown", "Other"]:
                assert "use_case" in f
            else:
                assert "use_case" not in f

    @pytest.mark.happy_path
    def test_empty_directory(self, temp_empty_dir, mock_st_error_warning):
        """
        Test processing an empty directory.
        """
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo(temp_empty_dir, include_options, llm_provider)
        assert result["repo_path"] == temp_empty_dir
        assert result["files"] == []
        assert result["languages"] == []
        assert result["error"] is None

    # --- Edge Case Tests ---

    @pytest.mark.edge_case
    def test_nonexistent_directory(self, mock_st_error_warning):
        """
        Test with a directory path that does not exist.
        """
        fake_path = "/tmp/this/path/does/not/exist"
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo(fake_path, include_options, llm_provider)
        assert result["error"] is not None
        assert result["files"] == []
        assert result["languages"] is None

    @pytest.mark.edge_case
    def test_path_is_file_not_directory(self, temp_repo_with_files, mock_st_error_warning):
        """
        Test with a path that is a file, not a directory.
        """
        _, file_paths = temp_repo_with_files
        file_path = next(iter(file_paths.values()))
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo(file_path, include_options, llm_provider)
        assert result["error"] is not None
        assert result["files"] == []
        assert result["languages"] is None

    @pytest.mark.edge_case
    def test_permission_error_on_directory(self, temp_repo_with_files, mock_st_error_warning, monkeypatch):
        """
        Simulate a permission error when accessing the directory.
        """
        repo_path, _ = temp_repo_with_files
        monkeypatch.setattr("os.listdir", mock.Mock(side_effect=PermissionError("No permission")))
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo(repo_path, include_options, llm_provider)
        assert result["error"] is not None
        assert result["files"] == []
        assert result["languages"] is None

    @pytest.mark.edge_case
    def test_file_too_large_for_token_count(self, temp_repo_with_files, mock_st_error_warning, monkeypatch):
        """
        Simulate a file that is too large for token counting.
        """
        repo_path, file_paths = temp_repo_with_files
        large_file = file_paths["large.txt"]
        # Patch os.path.getsize to return >5MB for this file
        orig_getsize = os.path.getsize

        def fake_getsize(path):
            if path == large_file:
                return 6 * 1024 * 1024  # 6MB
            return orig_getsize(path)

        monkeypatch.setattr("os.path.getsize", fake_getsize)
        include_options = {"token_count": True, "descriptions": False, "use_cases": False}
        llm_provider = "dummy"
        result = process_repo(repo_path, include_options, llm_provider)
        large_file_data = next(f for f in result["files"] if f["path"] == large_file)
        assert large_file_data["token_count"] == "File too large to count tokens"

    @pytest.mark.edge_case
    def test_file_not_found_during_token_count(self, temp_repo_with_files, mock_st_error_warning, monkeypatch):
        """
        Simulate a file being deleted between directory traversal and reading for token count.
        """
        repo_path, file_paths = temp_repo_with_files
        target_file = file_paths["main.py"]

        orig_open = open

        def fake_open(path, *args, **kwargs):
            if path == target_file:
                raise FileNotFoundError("Simulated missing file")
            return orig_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        include_options = {"token_count": True, "descriptions": False, "use_cases": False}
        llm_provider = "dummy"
        result = process_repo(repo_path, include_options, llm_provider)
        file_data = next(f for f in result["files"] if f["path"] == target_file)
        assert file_data["token_count"] == "File not found during count"

    @pytest.mark.edge_case
    def test_directory_path_is_empty_string(self, mock_st_error_warning):
        """
        Test with an empty string as the directory path.
        """
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo("", include_options, llm_provider)
        assert result["error"] is not None
        assert result["files"] == []
        assert result["languages"] is None

    @pytest.mark.edge_case
    def test_directory_path_is_not_a_string(self, mock_st_error_warning):
        """
        Test with a non-string directory path.
        """
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo(12345, include_options, llm_provider)
        assert result["error"] is not None
        assert result["files"] == []
        assert result["languages"] is None

    @pytest.mark.edge_case
    def test_traverse_directory_oserror(self, temp_repo_with_files, mock_st_error_warning, monkeypatch):
        """
        Simulate OSError during directory traversal.
        """
        repo_path, _ = temp_repo_with_files
        monkeypatch.setattr("streamlit_app.streamlit_app.traverse_directory", mock.Mock(side_effect=OSError("Simulated OSError")))
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo(repo_path, include_options, llm_provider)
        assert result["error"] is not None
        assert result["files"] == []
        assert result["languages"] is None

    @pytest.mark.edge_case
    def test_unexpected_exception_in_processing(self, temp_repo_with_files, mock_st_error_warning, monkeypatch):
        """
        Simulate an unexpected exception in the main processing loop.
        """
        repo_path, _ = temp_repo_with_files
        # Patch os.path.isfile to raise an exception
        monkeypatch.setattr("os.path.isfile", mock.Mock(side_effect=Exception("Unexpected error")))
        include_options = {"token_count": True, "descriptions": True, "use_cases": True}
        llm_provider = "dummy"
        result = process_repo(repo_path, include_options, llm_provider)
        assert result["error"] is not None
        assert result["files"] == []
        assert result["languages"] is None