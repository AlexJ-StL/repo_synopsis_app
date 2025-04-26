"""Testing file for the Streamlit repository synopsis application."""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, NamedTuple
from unittest.mock import patch, MagicMock

# Third party imports
import pytest
import streamlit as st

# Local application imports
# Ensure functions are imported *after* potential sys.path modifications if needed,
# though direct relative imports are usually preferred if structure allows.
# Assuming streamlit_app is adjacent or PYTHONPATH is set correctly
from streamlit_app.streamlit_app import (
    handle_directory_error,
    save_synopsis,
    # generate_synopsis, # Removed as the function structure changed
    log_event,
    main,
    traverse_directory,
    generate_directory_tree,
    get_file_language,
    get_llm_response,
    process_repo,
    summarize_text,
    RepoData, # Import the TypedDict
    generate_synopsis_text # Import the new text generation function
)

# --- Test Fixtures and Mocks ---

@pytest.fixture(autouse=True)
def suppress_streamlit_output(monkeypatch):
    """Automatically mock streamlit UI functions to prevent actual UI calls."""
    monkeypatch.setattr(st, "error", MagicMock())
    monkeypatch.setattr(st, "warning", MagicMock())
    monkeypatch.setattr(st, "info", MagicMock())
    monkeypatch.setattr(st, "success", MagicMock())
    monkeypatch.setattr(st, "progress", MagicMock(return_value=MagicMock())) # Mock progress bar object
    monkeypatch.setattr(st, "empty", MagicMock(return_value=MagicMock())) # Mock empty placeholder
    monkeypatch.setattr(st, "markdown", MagicMock())
    monkeypatch.setattr(st, "expander", MagicMock()) # Mock expander
    # Mock sidebar elements if needed for specific tests, but often mocking
    # the functions that *use* the sidebar values is sufficient.
    # monkeypatch.setattr(st.sidebar, "checkbox", MagicMock(return_value=True))
    # monkeypatch.setattr(st.sidebar, "selectbox", MagicMock(return_value="Groq"))
    # monkeypatch.setattr(st.sidebar, "text_input", MagicMock(return_value="."))


# Define a default filename used in save_synopsis tests
DEFAULT_SYNOPSIS_FILENAME = "synopsis_test.txt"

# --- Test Functions ---

def test_handle_directory_error_empty_path():
    """Test handle_directory_error with an empty path."""
    assert handle_directory_error("") is False


def test_handle_directory_error_valid_path(tmp_path: Path):
    """Test handle_directory_error with a valid path."""
    assert handle_directory_error(str(tmp_path)) is True


def test_handle_directory_error_nonexistent_path():
    """Test handle_directory_error with a nonexistent path."""
    assert handle_directory_error("/nonexistent/path/hopefully") is False


def test_handle_directory_error_os_error(tmp_path: Path):
    """Test handle_directory_error with an OSError during listdir."""
    with patch("os.listdir", side_effect=OSError("Simulated OSError")):
        assert handle_directory_error(str(tmp_path)) is False


def test_handle_directory_error_permission_error(tmp_path: Path):
    """Test handle_directory_error with a permission error during listdir."""
    with patch("os.listdir", side_effect=PermissionError("Permission denied")):
        assert handle_directory_error(str(tmp_path)) is False


def test_handle_directory_error_not_a_directory(tmp_path: Path):
    """Test handle_directory_error when the path is a file, not a directory."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("content")
    assert handle_directory_error(str(test_file)) is False


def test_save_synopsis_success(tmp_path: Path):
    """Test save_synopsis with a successful save."""
    # Mock handle_directory_error to always return True for this test
    with patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True):
        assert save_synopsis(str(tmp_path), "test content", DEFAULT_SYNOPSIS_FILENAME) is True
        assert (tmp_path / DEFAULT_SYNOPSIS_FILENAME).exists()
        assert (tmp_path / DEFAULT_SYNOPSIS_FILENAME).read_text() == "test content"


def test_save_synopsis_empty_content(tmp_path: Path):
    """Test save_synopsis when the content is empty."""
    # Mock handle_directory_error to always return True for this test
    with patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True):
        assert save_synopsis(str(tmp_path), "", DEFAULT_SYNOPSIS_FILENAME) is False


def test_save_synopsis_ioerror(tmp_path: Path):
    """Test save_synopsis with an IOError during open."""
    # Mock handle_directory_error to always return True for this test
    with patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True):
        with patch("builtins.open", side_effect=IOError("Simulated IOError")):
            assert save_synopsis(str(tmp_path), "Test", DEFAULT_SYNOPSIS_FILENAME) is False


def test_save_synopsis_invalid_dir(tmp_path: Path):
    """Test save_synopsis when the directory is invalid."""
    # Rely on the actual handle_directory_error check within save_synopsis
    assert save_synopsis("/nonexistent/path", "Test", DEFAULT_SYNOPSIS_FILENAME) is False


def test_log_event_success(tmp_path: Path):
    """Test log_event with a successful log."""
    log_event(str(tmp_path), "test message")
    log_file = tmp_path / "event_log.txt"
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as file:
        content = file.read()
        assert "test message" in content


def test_log_event_failure(tmp_path: Path):
    """Test log_event recovers gracefully from an IOError during open."""
    # We expect it to print an error, not raise the IOError
    with patch("builtins.open", side_effect=IOError("Simulated IO Error")):
        # Should not raise an exception
        log_event(str(tmp_path), "test message that fails to write")


# --- Tests for generate_synopsis_text (New Function) ---
# It's recommended to add tests for generate_synopsis_text similar to how
# the old generate_synopsis was tested, but passing RepoData as input.

@pytest.fixture
def sample_repo_data() -> RepoData:
    """Provides a sample RepoData structure for testing."""
    return {
        "repo_path": "/fake/repo",
        "files": [
            {
                "path": "/fake/repo/file1.py",
                "language": "Python",
                "token_count": 10,
                "description": "A python file.",
                "use_case": "Core Logic"
            },
            {
                "path": "/fake/repo/subdir/file2.js",
                "language": "JavaScript",
                "token_count": 25,
                "description": "Some javascript.",
                "use_case": "Utility"
            }
        ],
        "languages": ["Python", "JavaScript"],
        "error": None
    }

@patch("streamlit_app.streamlit_app.generate_directory_tree")
def test_generate_synopsis_text_all_options(
    mock_generate_tree: MagicMock,
    sample_repo_data: RepoData):
    """Test generate_synopsis_text includes all parts when requested."""
    mock_generate_tree.return_value = "Mock Tree"
    result = generate_synopsis_text(sample_repo_data, include_tree=True, directory_path="/fake/repo")

    assert "Languages used: Python, JavaScript" in result
    assert "## Directory Tree" in result
    assert "Mock Tree" in result
    assert "## File Details" in result
    assert "File: `/fake/repo/file1.py` (Python)" in result
    assert "Token Count:** 10" in result
    assert "Description:** A python file." in result
    assert "Use Case:** Core Logic" in result
    assert "File: `/fake/repo/subdir/file2.js` (JavaScript)" in result


@patch("streamlit_app.streamlit_app.generate_directory_tree")
def test_generate_synopsis_text_no_tree(
    mock_generate_tree: MagicMock,
    sample_repo_data: RepoData):
    """Test generate_synopsis_text excludes tree when requested."""
    result = generate_synopsis_text(sample_repo_data, include_tree=False, directory_path="/fake/repo")

    assert "## Directory Tree" not in result
    assert "Mock Tree" not in result
    mock_generate_tree.assert_not_called()
    assert "## File Details" in result # Details should still be present


def test_generate_synopsis_text_no_files() -> None:
    """Test generate_synopsis_text with empty files list."""
     # Create RepoData with no files
    empty_repo_data: RepoData = {
        "repo_path": "/fake/empty",
        "files": [],
        "languages": ["Python"], # Still has language info
        "error": None
    }
    with patch("streamlit_app.streamlit_app.generate_directory_tree", return_value="Mock Tree"):
         result = generate_synopsis_text(empty_repo_data, include_tree=True, directory_path="/fake/empty")

    assert "Languages used: Python" in result
    assert "## Directory Tree" in result
    assert "## File Details" not in result # No file details section


# --- Tests for Remaining Helper Functions ---

def test_traverse_directory(tmp_path: Path):
    """Test traverse_directory with files and subdirectories."""
    file1 = tmp_path / "test1.txt"
    file1.write_text("content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file2 = subdir / "test2.txt"
    file2.write_text("content")

    items = traverse_directory(str(tmp_path))
    # Normalize paths for reliable comparison
    expected_paths = {
        os.path.normpath(str(file1)),
        os.path.normpath(str(file2))
    }
    assert set(map(os.path.normpath, items)) == expected_paths


def test_traverse_directory_empty(tmp_path: Path):
    """Test traverse_directory with an empty directory."""
    result = traverse_directory(str(tmp_path))
    assert not result


def test_traverse_directory_error():
    """Test traverse_directory returns empty list on OSError."""
    with patch("os.walk", side_effect=OSError("Simulated OSError")):
        items = traverse_directory("/fake/path")
        assert items == [] # Should return empty list, not raise


def test_generate_directory_tree(tmp_path: Path):
    """Test generate_directory_tree basic structure."""
    file1 = tmp_path / "test1.txt"
    file1.write_text("content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file2 = subdir / "test2.txt"
    file2.write_text("content")
    empty_subdir = tmp_path / "empty_subdir"
    empty_subdir.mkdir()

    tree = generate_directory_tree(str(tmp_path))
    print(tree) # For debugging test failures
    assert "  test1.txt" in tree # File at root level
    assert "subdir/" in tree # Subdir name
    assert "  test2.txt" in tree # File in subdir
    assert "empty_subdir/" in tree # Empty subdir name
    # Check indentation levels implicitly by checking content order/presence


def test_generate_directory_tree_with_error():
    """Test generate_directory_tree returns error message on OSError."""
    with patch("os.walk", side_effect=OSError("Permission denied")):
        tree = generate_directory_tree("/fake/path")
        assert "Error generating tree: Permission denied" in tree


@patch("streamlit_app.streamlit_app.get_summarizer")
def test_summarize_text_success(mock_get_summarizer: MagicMock):
    """Test the text summarization function success path."""
    # Mock the pipeline object and its call
    mock_pipeline = MagicMock(return_value=[{"summary_text": "Short summary."}])
    mock_get_summarizer.return_value = mock_pipeline

    text = ("This is a long piece of text that needs to be summarized. "
            "It contains multiple sentences and should be made shorter. " * 5)
    summary = summarize_text(text)

    assert summary == "Short summary."
    mock_pipeline.assert_called_once()


@patch("streamlit_app.streamlit_app.get_summarizer")
def test_summarize_text_too_short(mock_get_summarizer: MagicMock):
    """Test summarize_text returns empty string for short text."""
    text = "This text is too short."
    summary = summarize_text(text)
    assert summary == "" # Changed behavior: returns empty string now
    mock_get_summarizer.assert_not_called()


@patch("streamlit_app.streamlit_app.get_summarizer")
def test_summarize_text_pipeline_error(mock_get_summarizer: MagicMock):
    """Test summarize_text returns empty string on pipeline error."""
    mock_pipeline = MagicMock(side_effect=Exception("Pipeline crashed"))
    mock_get_summarizer.return_value = mock_pipeline

    text = "This text will cause an error during summarization." * 5
    summary = summarize_text(text)
    assert summary == ""


@patch("streamlit_app.streamlit_app.get_summarizer")
def test_summarize_text_unexpected_output(mock_get_summarizer: MagicMock):
    """Test summarize_text returns empty string on unexpected pipeline output."""
    mock_pipeline = MagicMock(return_value="just a string") # Invalid output
    mock_get_summarizer.return_value = mock_pipeline

    text = "This text gets an unexpected output format." * 5
    summary = summarize_text(text)
    assert summary == ""


def test_get_file_language_known():
    """Test get_file_language with known extensions."""
    assert get_file_language("my_file.py") == "Python"
    assert get_file_language("script.js") == "JavaScript"
    assert get_file_language("Data.JSON") == "JSON" # Test case insensitivity
    assert get_file_language("README.md") == "Markdown"
    assert get_file_language("Dockerfile") == "Dockerfile" # Test basename


def test_get_file_language_unknown():
    """Test get_file_language with unknown extensions."""
    assert get_file_language("mystery.xyz") == "Other" # Default changed to 'Other'
    assert get_file_language("no_extension") == "Other"


@patch("streamlit_app.streamlit_app.summarize_text")
def test_get_llm_response_success(mock_summarize: MagicMock, tmp_path: Path):
    """Test get_llm_response success path."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text content")
    mock_summarize.return_value = "Summarized text"

    description, use_case = get_llm_response(str(file_path), "Groq")

    assert description == "Summarized text"
    assert use_case == "Core Logic/Component" # Default use case
    mock_summarize.assert_called_once_with("Sample text content")


def test_get_llm_response_file_not_found():
    """Test get_llm_response when file doesn't exist."""
    description, use_case = get_llm_response("/non/existent/file.py", "Groq")
    assert "Error: File not found" in description
    assert "Error: File not found" in use_case


def test_get_llm_response_read_error(tmp_path: Path):
    """Test get_llm_response with an error during file read."""
    file_path = tmp_path / "test.txt"
    # No write_text to simulate potential read issue, or mock open
    with patch("builtins.open", side_effect=IOError("Simulated Read Error")):
        description, use_case = get_llm_response(str(file_path), "Groq")
        assert "Error reading file: Simulated Read Error" in description
        assert "Error: Simulated Read Error" in use_case


@patch("streamlit_app.streamlit_app.summarize_text")
def test_get_llm_response_summarize_error(mock_summarize: MagicMock, tmp_path: Path):
    """Test get_llm_response when summarization returns empty string (error)."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text content")
    mock_summarize.return_value = "" # Simulate summarization error/empty result

    description, use_case = get_llm_response(str(file_path), "Groq")

    assert description == "" # Should pass through the empty description
    assert use_case == "Core Logic/Component"
    mock_summarize.assert_called_once()


# --- process_repo Tests ---

# Define default options used in process_repo tests
DEFAULT_INCLUDE_OPTIONS = {
    "token_count": True,
    "descriptions": True,
    "use_cases": True
}

@patch("streamlit_app.streamlit_app.get_llm_response")
@patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
def test_process_repo_success(
    mock_handle_dir: MagicMock,
    mock_get_llm: MagicMock,
    tmp_path: Path):
    """Test process_repo successfully processes a file."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "file1.py").write_text("print('hello world')") # 2 words
    mock_get_llm.return_value = ("Mock description", "Mock use case")

    repo_data = process_repo(str(repo_path), DEFAULT_INCLUDE_OPTIONS, "Groq")

    mock_handle_dir.assert_called_once_with(str(repo_path))
    mock_get_llm.assert_called_once()
    assert repo_data.get("error") is None
    assert len(repo_data["files"]) == 1
    file_info = repo_data["files"][0]
    # Use .get() for potentially missing keys or assert their presence if guaranteed by test setup
    assert file_info.get("language") == "Python"
    assert file_info.get("token_count") == 2
    assert file_info.get("description") == "Mock description"
    assert file_info.get("use_case") == "Mock use case"
    assert repo_data.get("languages") == ["Python"]


@patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
def test_process_repo_no_files(mock_handle_dir: MagicMock, tmp_path: Path):
    """Test process_repo with an empty directory."""
    repo_path = tmp_path / "empty_repo"
    repo_path.mkdir()

    repo_data = process_repo(str(repo_path), DEFAULT_INCLUDE_OPTIONS, "Groq")

    mock_handle_dir.assert_called_once_with(str(repo_path))
    assert repo_data.get("error") is None
    assert len(repo_data["files"]) == 0
    assert repo_data["languages"] == []


def test_process_repo_invalid_dir():
    """Test process_repo when the initial directory check fails."""
    # handle_directory_error is mocked globally to return True by default,
    # so we patch it specifically for this test to return False.
    with patch("streamlit_app.streamlit_app.handle_directory_error", return_value=False) as mock_handle_dir:
        repo_data = process_repo("/invalid/path", DEFAULT_INCLUDE_OPTIONS, "Groq")

        mock_handle_dir.assert_called_once_with("/invalid/path")
        # Check error field safely
        error_msg = repo_data.get("error")
        assert error_msg is not None
        assert "Invalid or inaccessible directory" in error_msg
        assert repo_data.get("files") == []
        assert repo_data.get("languages") is None


@patch("streamlit_app.streamlit_app.traverse_directory", side_effect=Exception("Traversal Error"))
@patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
def test_process_repo_traversal_error(mock_handle_dir: MagicMock, mock_traverse: MagicMock, tmp_path: Path):
    """Test process_repo when os.walk (via traverse_directory) fails."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir() # Directory needs to exist for initial check

    repo_data = process_repo(str(repo_path), DEFAULT_INCLUDE_OPTIONS, "Groq")

    mock_handle_dir.assert_called_once_with(str(repo_path))
    mock_traverse.assert_called_once_with(str(repo_path))
    # Check error field safely
    error_msg = repo_data.get("error")
    assert error_msg is not None
    assert "Unexpected processing error: Traversal Error" in error_msg
    assert repo_data.get("files") == []
    assert repo_data.get("languages") is None


@patch("builtins.open", side_effect=IOError("Read Error"))
@patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
def test_process_repo_file_read_error_token(mock_handle_dir: MagicMock, mock_open: MagicMock, tmp_path: Path):
    """Test process_repo handles file read error during token counting."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "file1.py").touch() # Create file

    options = {"token_count": True, "descriptions": False, "use_cases": False}
    repo_data = process_repo(str(repo_path), options, "Groq")

    assert repo_data.get("error") is None # No error at the repo level
    assert len(repo_data["files"]) == 1
    file_info = repo_data["files"][0]
    # Use .get() for potentially missing keys
    assert file_info.get("description") == "LLM Error description"
    assert file_info.get("use_case") == "LLM Error use case"
    # Assert required keys are present
    assert file_info.get("language") == "Python"
    # builtins.open would be called by process_repo for token count


@patch("streamlit_app.streamlit_app.get_llm_response", side_effect=Exception("LLM Error"))
@patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
def test_process_repo_llm_error(mock_handle_dir: MagicMock, mock_get_llm: MagicMock, tmp_path: Path):
    """Test process_repo continues if get_llm_response fails for one file."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "file1.py").write_text("print('hello')")

    # We expect get_llm_response to be called, raise an error, but the overall
    # process_repo should still complete and return data for the file.
    # The error handling is *within* get_llm_response itself now.
    # Let's redefine get_llm_response mock to return error strings
    mock_get_llm.side_effect = None # Clear previous side_effect
    mock_get_llm.return_value = ("LLM Error description", "LLM Error use case")

    repo_data = process_repo(str(repo_path), DEFAULT_INCLUDE_OPTIONS, "Groq")

    assert repo_data.get("error") is None # No error at the repo level
    assert len(repo_data["files"]) == 1
    file_info = repo_data["files"][0]
    error_msg = repo_data.get("error")
    assert error_msg is not None
    assert "Unexpected processing error: LLM error" in error_msg
    assert file_info.get("description") == "LLM Error description" # Error captured by get_llm_response
    assert file_info.get("use_case") == "LLM Error use case"


# --- main Function Test (Simplified) ---

# Note: Testing the full Streamlit UI flow with button clicks, session state,
# and dynamic updates typically requires streamlit.testing.AppTest.
# This test focuses on the core logic path triggered by the button press.

@patch("streamlit_app.streamlit_app.st.button", return_value=True) # Simulate button press
@patch("streamlit_app.streamlit_app.st.session_state", new_callable=MagicMock) # Mock session state
@patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True) # Assume valid dir
@patch("streamlit_app.streamlit_app.process_repo") # Mock the core processing
@patch("streamlit_app.streamlit_app.save_synopsis", return_value=True) # Assume saving works
@patch("streamlit_app.streamlit_app.log_event") # Mock logging
@patch("builtins.open") # Mock file opening for JSON dump and download
@patch("streamlit_app.streamlit_app.json.dump") # Mock JSON writing
@patch("streamlit_app.streamlit_app.generate_synopsis_text", return_value="Synopsis Text") # Mock text gen
@patch("streamlit_app.streamlit_app.st.download_button") # Mock download buttons
def test_main_processing_logic(
    mock_download_button: MagicMock,
    mock_gen_text: MagicMock,
    mock_json_dump: MagicMock,
    mock_open: MagicMock,
    mock_log: MagicMock,
    mock_save: MagicMock,
    mock_process: MagicMock,
    mock_handle_dir: MagicMock,
    mock_session_state: MagicMock,
    mock_st_button: MagicMock,
    tmp_path: Path # Use tmp_path for base directory
    ):
    """Test the main logic path after the 'Generate' button is pressed."""

    # --- Setup Mocks and State ---
    base_dir = str(tmp_path)
    repo_name = "my_repo"
    full_repo_path = os.path.join(base_dir, repo_name)
    os.makedirs(full_repo_path, exist_ok=True) # Create dummy repo dir

    # Simulate UI state needed *after* button press
    # We need to mock access to st.session_state AFTER the button is checked
    mock_session_state.repo_select = [repo_name] # Simulate repo selected

    # Mock return value for process_repo
    mock_process.return_value = RepoData(
        repo_path=full_repo_path, files=[], languages=[], error=None
    )

    # --- Mock Sidebar/Config values needed by main ---
    # Use patch.dict for st.session_state if mocking individual keys/widgets
    # Or directly set attributes on the MagicMock as done above for repo_select
    with patch.dict(st.session_state, {
        'base_dir': base_dir, # Simulate text input value
        'inc_tree': True,
        'inc_desc': True,
        'inc_token': True,
        'inc_use': True,
        'llm_select': 'Groq',
        'repo_select': [repo_name] # Ensure it's set before main() is called
    }):
        # --- Execute ---
        main() # Call the main function

    # --- Assertions ---
    # Check if core functions were called
    mock_handle_dir.assert_called() # Called for base_dir validation
    mock_st_button.assert_called_with("Generate", key="generate_button", type="primary")
    mock_process.assert_called_once()
    # Check args of process_repo if necessary:
    call_args, call_kwargs = mock_process.call_args
    assert call_args[0] == full_repo_path # Check repo path arg
    assert call_args[1]['descriptions'] is True # Check options passed

    # Check if output generation/saving was attempted
    mock_gen_text.assert_called_once()
    mock_save.assert_called() # Called for the .md file
    mock_json_dump.assert_called_once() # Called for the .json file
    mock_log.assert_called() # Check if logging happened

    # Check download buttons were called
    assert mock_download_button.call_count == 2 # One for md, one for json

