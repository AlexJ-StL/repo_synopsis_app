import pytest
import shutil  # Import shutil
import os
import streamlit as st
from streamlit_app.streamlit_app import (
    traverse_directory,
    generate_directory_tree,
    get_file_language,
    save_synopsis,
    generate_synopsis,
    get_llm_response,
    log_event
)


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


def test_save_synopsis_empty_directory(tmpdir, monkeypatch):
    def mock_st_error(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    save_synopsis("", "test synopsis")


def test_save_synopsis_os_error(tmpdir, monkeypatch):
    def mock_makedirs(path, exist_ok=False):
        raise OSError("Mock OSError")
    monkeypatch.setattr(os, 'makedirs', mock_makedirs)
    def mock_st_error(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    save_synopsis(tmpdir, "test synopsis")


def test_save_synopsis_io_error(tmpdir, monkeypatch):
    def mock_open(*args, **kwargs):
        raise IOError("Mock IOError")
    monkeypatch.setattr("builtins.open", mock_open)
    def mock_st_error(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    save_synopsis(tmpdir, "test synopsis")


# --- Test generate_synopsis ---
# More complex, requires setting up a directory and mocking st functions
def test_generate_synopsis_empty(tmpdir, monkeypatch):
    # Mock st.write and st.error to avoid Streamlit dependencies
    def mock_write(x): return None  # noqa: E731
    def mock_error(x): return None  # noqa: E731
    monkeypatch.setattr(st, 'write', mock_write)
    monkeypatch.setattr(st, 'error', mock_error)

    generate_synopsis(str(tmpdir), False, False, False, False, "Groq")
    # Assert nothing is written to stdout/stderr (or check logs)


# Add tests for generate_synopsis with different directory structures & options
def test_generate_synopsis_with_all_options(tmpdir, monkeypatch):
    # Mock streamlit functions
    def mock_write(x): return None
    def mock_error(x): return None
    def mock_success(x): return None
    monkeypatch.setattr(st, 'write', mock_write)
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr(st, 'success', mock_success)

    # Create test files
    file1 = tmpdir.join("test.py")
    file1.write("print('hello')")
    subdir = tmpdir.mkdir("subdir")
    file2 = subdir.join("test.js")
    file2.write("console.log('hello')")

    synopsis = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )

    assert synopsis is not None
    assert "Directory Tree" in synopsis
    assert "test.py" in synopsis
    assert "test.js" in synopsis
    assert "Token Count" in synopsis
    assert "Description" in synopsis
    assert "Use Case" in synopsis

def test_generate_synopsis_invalid_path(monkeypatch):
    # Mock streamlit functions
    def mock_error(x): return None
    monkeypatch.setattr(st, 'error', mock_error)

    result = generate_synopsis(
        "invalid/path",
        True, True, True, True,
        "Groq"
    )
    assert result is None

def test_log_event(tmpdir):
    log_message = "Test log message"
    log_event(str(tmpdir), log_message)

    log_file = os.path.join(str(tmpdir), "event_log.txt")
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        content = f.read()
        assert log_message in content

def test_log_event_error(tmpdir, monkeypatch):
    def mock_open(*args, **kwargs):
        raise IOError("Mock IOError")
    monkeypatch.setattr("builtins.open", mock_open)

    def mock_st_write(x): return None
    monkeypatch.setattr(st, 'write', mock_st_write)

    # Should not raise an exception
    log_event(str(tmpdir), "Test message")

def test_save_synopsis_with_custom_directory(tmpdir):
    synopsis_content = "Test synopsis"
    custom_dir = tmpdir.mkdir("custom")

    save_synopsis(str(tmpdir), synopsis_content, str(custom_dir))

    saved_file = custom_dir.join("repo_synopsis.md")
    assert saved_file.exists()
    assert saved_file.read() == synopsis_content

def test_generate_directory_tree_nested(tmpdir):
    # Create a nested directory structure
    dir1 = tmpdir.mkdir("dir1")
    dir2 = dir1.mkdir("dir2")
    file1 = dir2.join("file1.txt")
    file1.write("content")

    tree = generate_directory_tree(str(tmpdir))
    assert "dir1" in tree
    assert "dir2" in tree
    assert "file1.txt" in tree

def test_traverse_directory_with_hidden_files(tmpdir):
    # Create some regular and hidden files
    file1 = tmpdir.join("visible.txt")
    file1.write("content")
    hidden = tmpdir.join(".hidden")
    hidden.write("hidden content")

    items = traverse_directory(str(tmpdir))
    assert str(file1) in items
    assert str(hidden) in items

def test_get_file_language_additional_types():
    assert get_file_language("test.tsx") == "TypeScript"
    assert get_file_language("test.jsx") == "JavaScript"
    assert get_file_language("test.cpp") == "C/C++"
    assert get_file_language("test.rs") == "Rust"
    assert get_file_language("test.go") == "Go"

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


def test_get_llm_response():
    description, use_case = get_llm_response("test.py", "Groq")
    assert description == "Groq description placeholder"
    assert use_case == "Groq use case placeholder"

    description, use_case = get_llm_response("test.py", "Cerberas")
    assert description == "Cerberas description placeholder"
    assert use_case == "Cerberas use case placeholder"

    description, use_case = get_llm_response("test.py", "SombaNova")
    assert description == "SombaNova description placeholder"
    assert use_case == "SombaNova use case placeholder"

    description, use_case = get_llm_response("test.py", "Gemini")
    assert description == "Gemini description placeholder"
    assert use_case == "Gemini use case placeholder"

    description, use_case = get_llm_response("test.py", "Invalid")
    assert description == "Default description placeholder"
    assert use_case == "Default use case placeholder"
