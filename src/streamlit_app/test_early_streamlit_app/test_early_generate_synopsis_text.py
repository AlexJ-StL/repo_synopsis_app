import pytest
from unittest.mock import patch
from typing import cast, Dict, List, Optional

# Import the function under test
from streamlit_app.streamlit_app import generate_synopsis_text, RepoData, FileData

@pytest.mark.usefixtures("mock_os_walk")
class TestGenerateSynopsisText:
    """Unit tests for generate_synopsis_text in streamlit_app.streamlit_app."""

    @pytest.fixture
    def mock_os_walk(self):
        """Patch os.walk for directory tree generation."""
        with patch("streamlit_app.streamlit_app.os.walk") as mock_walk:
            # Default: empty directory
            mock_walk.return_value = [
                ("/repo", ["subdir"], ["file1.py", "file2.txt"]),
                ("/repo/subdir", [], ["file3.md"])
            ]
            yield mock_walk

    @pytest.fixture
    def minimal_repo_data(self):
        """A minimal valid RepoData mock."""
        return {
            "repo_path": "/repo",
            "files": [],
            "languages": [],
            "error": None
        }

    @pytest.fixture
    def full_repo_data(self):
        """A full-featured RepoData mock with multiple files and languages."""
        return {
            "repo_path": "/repo",
            "files": [
                {
                    "path": "main.py",
                    "language": "Python",
                    "token_count": 123,
                    "description": "Main entry point.",
                    "use_case": "CLI"
                },
                {
                    "path": "utils/helper.js",
                    "language": "JavaScript",
                    "token_count": 45,
                    "description": "Helper functions.",
                    "use_case": "Web"
                }
            ],
            "languages": ["Python", "JavaScript"],
            "error": None
        }

    @pytest.fixture
    def file_with_missing_fields(self):
        """A file entry with missing optional fields."""
        return {
            "repo_path": "/repo",
            "files": [
                {
                    "path": "README.md",
                    "language": "Markdown"
                    # No token_count, description, or use_case
                }
            ],
            "languages": ["Markdown"],
            "error": None
        }

    # --- Happy Path Tests ---

    @pytest.mark.happy
    def test_languages_and_tree_and_files(self, full_repo_data, mock_os_walk):
        """
        Test synopsis generation with languages, directory tree, and multiple files.
        """
        result = generate_synopsis_text(cast(RepoData, full_repo_data), include_tree=True, directory_path="/repo")
        # Check languages
        assert "Languages used: Python, JavaScript" in result
        # Check directory tree
        assert "## Directory Tree" in result
        assert "subdir/" in result
        assert "file1.py" in result
        assert "file2.txt" in result
        assert "file3.md" in result
        # Check file details
        assert "## File Details" in result
        assert "### File: `main.py` (Python)" in result
        assert "- **Token Count:** 123" in result
        assert "- **Description:** Main entry point." in result
        assert "- **Use Case:** CLI" in result
        assert "### File: `utils/helper.js` (JavaScript)" in result
        assert "- **Token Count:** 45" in result
        assert "- **Description:** Helper functions." in result
        assert "- **Use Case:** Web" in result

    @pytest.mark.happy
    def test_languages_no_tree_with_files(self, full_repo_data, mock_os_walk):
        """
        Test synopsis generation with languages and files, but no directory tree.
        """
        result = generate_synopsis_text(cast(RepoData, full_repo_data), include_tree=False, directory_path="/repo")
        assert "Languages used: Python, JavaScript" in result
        assert "## Directory Tree" not in result
        assert "## File Details" in result
        assert "main.py" in result
        assert "utils/helper.js" in result

    @pytest.mark.happy
    def test_no_languages_with_tree_and_files(self, full_repo_data, mock_os_walk):
        """
        Test synopsis generation with no languages, but with tree and files.
        """
        data = dict(full_repo_data)
        data.pop("languages", None)
        result = generate_synopsis_text(cast(RepoData, data), include_tree=True, directory_path="/repo")
        assert "Languages used:" not in result
        assert "## Directory Tree" in result
        assert "## File Details" in result

    @pytest.mark.happy
    def test_no_files_with_languages_and_tree(self, minimal_repo_data, mock_os_walk):
        """
        Test synopsis generation with languages and tree, but no files.
        """
        data = dict(minimal_repo_data)
        data["languages"] = ["Python"]
        result = generate_synopsis_text(cast(RepoData, data), include_tree=True, directory_path="/repo")
        assert "Languages used: Python" in result
        assert "## Directory Tree" in result
        assert "## File Details" not in result

    @pytest.mark.happy
    def test_file_with_missing_optional_fields(self, file_with_missing_fields, mock_os_walk):
        """
        Test file entry with missing token_count, description, and use_case.
        """
        result = generate_synopsis_text(cast(RepoData, file_with_missing_fields), include_tree=False, directory_path="/repo")
        assert "Languages used: Markdown" in result
        assert "## File Details" in result
        assert "### File: `README.md` (Markdown)" in result
        # Should not contain token count, description, or use case
        assert "- **Token Count:**" not in result
        assert "- **Description:**" not in result
        assert "- **Use Case:**" not in result

    # --- Edge Case Tests ---

    @pytest.mark.edge
    def test_empty_repo_data(self, minimal_repo_data, mock_os_walk):
        """
        Test with completely empty repo data (no languages, no files).
        """
        result = generate_synopsis_text(cast(RepoData, minimal_repo_data), include_tree=False, directory_path="/repo")
        assert result.strip() == ""  # Should be empty

    @pytest.mark.edge
    def test_empty_files_list(self, minimal_repo_data, mock_os_walk):
        """
        Test with empty files list but with languages.
        """
        data = dict(minimal_repo_data)
        data["languages"] = ["Python"]
        # Cast the dict to RepoData
        result = generate_synopsis_text(cast(RepoData, data), include_tree=False, directory_path="/repo")
        assert "Languages used: Python" in result
        assert "## File Details" not in result

    @pytest.mark.edge
    def test_file_with_token_count_as_string(self, mock_os_walk):
        """
        Test file entry where token_count is a string (e.g., error message).
        """
        repo_data = {
            "repo_path": "/repo",
            "files": [
                {
                    "path": "broken.py",
                    "language": "Python",
                    "token_count": "Error: Could not count tokens"
                }
            ],
            "languages": ["Python"],
            "error": None
        }
        result = generate_synopsis_text(cast(RepoData, repo_data), include_tree=False, directory_path="/repo")
        assert "- **Token Count:** Error: Could not count tokens" in result

    @pytest.mark.edge
    def test_file_with_extra_unexpected_fields(self, mock_os_walk):
        """
        Test file entry with extra, unexpected fields (should be ignored).
        """
        repo_data = {
            "repo_path": "/repo",
            "files": [
                {
                    "path": "foo.py",
                    "language": "Python",
                    "token_count": 10,
                    "extra_field": "should be ignored"
                }
            ],
            "languages": ["Python"],
            "error": None
        }
        result = generate_synopsis_text(cast(RepoData, repo_data), include_tree=False, directory_path="/repo")
        assert "foo.py" in result
        assert "should be ignored" not in result

    @pytest.mark.edge
    def test_directory_tree_oserror(self, full_repo_data):
        """
        Test that an OSError in directory tree generation is handled gracefully.
        """
        with patch("streamlit_app.streamlit_app.os.walk", side_effect=OSError("Permission denied")):
            # Cast the dict to RepoData
            result = generate_synopsis_text(cast(RepoData, full_repo_data), include_tree=True, directory_path="/repo")
            assert "Error generating tree: Permission denied" in result

    @pytest.mark.edge
    def test_file_with_missing_path_and_language(self, mock_os_walk):
        """
        Test file entry missing both path and language (should default to 'N/A').
        """
        repo_data = {
            "repo_path": "/repo",
            "files": [
                {
                    "token_count": 5
                }
            ],
            "languages": [],
            "error": None
        }
        result = generate_synopsis_text(cast(RepoData, repo_data), include_tree=False, directory_path="/repo")
        assert "### File: `N/A` (N/A)" in result
        assert "- **Token Count:** 5" in result

    @pytest.mark.edge
    def test_none_languages_and_files(self, mock_os_walk):
        """
        Test with languages and files set to None.
        """
        repo_data = {
            "repo_path": "/repo",
            "files": None,
            "languages": None,
            "error": None
        }
        result = generate_synopsis_text(cast(RepoData, repo_data), include_tree=False, directory_path="/repo")
        assert result.strip() == ""

    @pytest.mark.edge
    def test_files_key_missing(self, minimal_repo_data, mock_os_walk):
        """
        Test with 'files' key missing from repo_data.
        """
        data = dict(minimal_repo_data)
        data.pop("files", None)
        # Cast the dict to RepoData
        result = generate_synopsis_text(cast(RepoData, data), include_tree=False, directory_path="/repo")
        assert result.strip() == ""

    @pytest.mark.edge
    def test_languages_key_missing(self, minimal_repo_data, mock_os_walk):
        """
        Test with 'languages' key missing from repo_data.
        """
        data = dict(minimal_repo_data)
        data.pop("languages", None)
        # Cast is already present here, which is good.
        result = generate_synopsis_text(cast(RepoData, data), include_tree=False, directory_path="/repo")
        assert "Languages used:" not in result