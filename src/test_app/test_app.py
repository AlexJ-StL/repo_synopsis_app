import pytest
import shutil  # Import shutil
from streamlit_app.streamlit_app import (
    traverse_directory,
    generate_directory_tree,
    get_file_language,
    save_synopsis,
    generate_synopsis
)
import streamlit as st


# --- Test traverse_directory ---
def test_traverse_directory_empty(tmpdir):
    assert traverse_directory(tmpdir) == []


def test_traverse_directory_simple(tmpdir):
    # Create some files and directories
    file1 = tmpdir.join("file1.txt")
    file1.write("content")
    subdir = tmpdir.mkdir("subdir")
    file2 = subdir.join("file2.py")
    file2.write("print('hello')")

    expected_paths = sorted([
        str(file1),
        str(subdir),
        str(file2),
    ])
    actual_paths = sorted(traverse_directory(tmpdir))
    assert actual_paths == expected_paths


# --- Test generate_directory_tree ---
def test_generate_directory_tree_empty(tmpdir):
    assert generate_directory_tree(tmpdir) == ""


def test_generate_directory_tree_simple(tmpdir):
    file1 = tmpdir.join("file1.txt")
    file1.write("content")
    subdir = tmpdir.mkdir("subdir")
    file2 = subdir.join("file2.py")
    file2.write("print('hello')")

    expected_tree = "- file1.txt\\n- subdir\\n  - file2.py\\n"
    assert generate_directory_tree(tmpdir) == expected_tree


# --- Test get_file_language ---
def test_get_file_language_known():
    assert get_file_language("test.py") == "Python"
    assert get_file_language("test.js") == "JavaScript"
    assert get_file_language("test.md") == "Markdown"


def test_get_file_language_unknown():
    assert get_file_language("test.xyz") == "Unknown"


def test_get_file_language_no_extension():
    assert get_file_language("test") == "Unknown"


# --- Test save_synopsis ---
def test_save_synopsis_valid(tmpdir):
    synopsis_content = "This is a test synopsis."
    save_synopsis(tmpdir, synopsis_content)
    saved_file = tmpdir.join("repo_synopsis.md")
    assert saved_file.read() == synopsis_content


# --- Test generate_synopsis ---
# More complex, requires setting up a directory and mocking st functions
def test_generate_synopsis_empty(tmpdir, monkeypatch):
    # Mock st.write and st.error to avoid Streamlit dependencies
    mock_write = lambda x: None  # noqa: E731
    mock_error = lambda x: None  # noqa: E731
    monkeypatch.setattr(st, 'write', mock_write)
    monkeypatch.setattr(st, 'error', mock_error)

    generate_synopsis(str(tmpdir), False, False, False, False, "Groq")
    # Assert nothing is written to stdout/stderr (or check logs)


# Add tests for generate_synopsis with different directory structures & options
# --- Test Fixtures ---
@pytest.fixture(autouse=True)
def cleanup_temp_files(tmpdir):
    """
    This fixture automatically cleans up any files created in the tmpdir
    after each test function has completed.
    """
    yield  # This allows the test function to run
    # Cleanup: Remove any created files/directories
    try:
        shutil.rmtree(tmpdir)  # Use shutil.rmtree to remove directories
    except OSError:
        pass  # Ignore errors if the directory can't be removed
