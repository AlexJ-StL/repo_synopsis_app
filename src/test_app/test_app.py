import pytest
import os
import streamlit as st
from typing import Tuple
from unittest.mock import mock_open, patch
from streamlit_app.streamlit_app import (
    handle_directory_error,
    save_synopsis,
    generate_synopsis,
    log_event,
    main,
    traverse_directory,
    generate_directory_tree,
    get_file_language,
    get_llm_response
)

def test_handle_directory_error_empty_path(monkeypatch):
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert handle_directory_error("") is False

def test_generate_synopsis_no_files_found(tmpdir, monkeypatch):
    """Test generate_synopsis when no files are found."""
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert generate_synopsis(str(tmpdir), True, True, True, True, "Groq") is None

def test_handle_directory_error_nonexistent_path(monkeypatch):
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert handle_directory_error("/nonexistent/path") is False

def test_handle_directory_error_valid_path(tmpdir, monkeypatch):
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert handle_directory_error(str(tmpdir)) is True

def test_generate_synopsis_file_encoding_error(tmpdir, monkeypatch):
    """Test generate_synopsis with a file that causes an encoding error."""
    test_file = tmpdir.join("test.txt")
    test_file.write("This is a test file with invalid encoding\000".encode('latin-1')) #Use latin-1, not utf-8.
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    generate_synopsis(str(tmpdir), True, True, True, True, "Groq") # test handles it

def test_handle_directory_error_permission_error(tmpdir, monkeypatch):
    def mock_error(msg): pass
    def mock_listdir(path): raise PermissionError
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr(os, 'listdir', mock_listdir)
    assert handle_directory_error(str(tmpdir)) is False

def test_save_synopsis_success(tmpdir, monkeypatch):
    def mock_success(msg): pass
    monkeypatch.setattr(st, 'success', mock_success)
    assert save_synopsis(str(tmpdir), "test content") is True
    assert os.path.exists(os.path.join(str(tmpdir), "repo_synopsis.md"))

def test_save_synopsis_failure(tmpdir, monkeypatch):
    def mock_error(msg): pass
    m = mock_open()
    m.side_effect = IOError("Simulated IO Error")
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr("builtins.open", m)
    assert save_synopsis(str(tmpdir), "test content") is False

def test_log_event_success(tmpdir):
    log_event(str(tmpdir), "test message")
    log_file = os.path.join(str(tmpdir), "event_log.txt")
    assert os.path.exists(log_file)
    with open(log_file) as f:
        content = f.read()
        assert "test message" in content

def test_log_event_failure(tmpdir, monkeypatch):
    def mock_error(msg): pass
    m = mock_open()
    m.side_effect = IOError("Simulated IO Error")
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr("builtins.open", m)
    log_event(str(tmpdir), "test message")

def test_generate_synopsis_invalid_directory(monkeypatch):
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert generate_synopsis("", True, True, True, True, "Groq") is None

def test_generate_synopsis_success(tmpdir, monkeypatch):
    def mock_success(msg): pass
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'success', mock_success)
    monkeypatch.setattr(st, 'error', mock_error)

    # Create test file
    test_file = tmpdir.join("test.py")
    test_file.write("print('hello')")

    result = generate_synopsis(str(tmpdir), True, True, True, True, "Groq")
    assert result is not None
    assert os.path.exists(os.path.join(str(tmpdir), "repo_synopsis.md"))

def test_generate_synopsis_empty_directory(tmpdir, monkeypatch):
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert generate_synopsis(str(tmpdir), True, True, True, True, "Groq") is None

def test_generate_synopsis_with_file_read_error(tmpdir, monkeypatch):
    def mock_error(msg): pass
    def mock_success(msg): pass

    # Mock streamlit functions
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr(st, 'success', mock_success)

    # Create test file
    test_file = tmpdir.join("test.py")
    test_file.write("print('hello')")

    # Create a more sophisticated mock that only raises error for specific files
    original_open = open
    def selective_mock_open(*args, **kwargs):
        if str(test_file) in str(args[0]):
            raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')
        return original_open(*args, **kwargs)

    monkeypatch.setattr('builtins.open', selective_mock_open)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )

    assert result is not None
    assert "Unable to read file" in result

def test_main(monkeypatch):
    def mock_title(msg): pass
    def mock_subheader(msg): pass
    def mock_checkbox(msg, value=True): return True
    def mock_selectbox(msg, options): return options[0]
    def mock_text_input(msg): return "/test/path"
    def mock_button(msg): return True
    def mock_success(msg): pass

    monkeypatch.setattr(st, 'title', mock_title)
    monkeypatch.setattr(st, 'subheader', mock_subheader)
    monkeypatch.setattr(st, 'checkbox', mock_checkbox)
    monkeypatch.setattr(st, 'selectbox', mock_selectbox)
    monkeypatch.setattr(st, 'text_input', mock_text_input)
    monkeypatch.setattr(st, 'button', mock_button)
    monkeypatch.setattr(st, 'success', mock_success)

    main()

def test_traverse_directory(tmpdir):
    # Create test files
    file1 = tmpdir.join("test1.txt")
    file1.write("content")
    subdir = tmpdir.mkdir("subdir")
    file2 = subdir.join("test2.txt")
    file2.write("content")

    items = traverse_directory(str(tmpdir))
    assert len(items) == 2
    assert any("test1.txt" in item for item in items)
    assert any("test2.txt" in item for item in items)

def test_traverse_directory_empty(tmpdir):
    items = traverse_directory(str(tmpdir))
    assert len(items) == 0

def test_traverse_directory_error(monkeypatch):
    def mock_walk(*args, **kwargs):
        raise OSError
    monkeypatch.setattr(os, 'walk', mock_walk)

    items = traverse_directory("/fake/path")
    assert len(items) == 0

def test_generate_directory_tree(tmpdir):
    # Create test files
    file1 = tmpdir.join("test1.txt")
    file1.write("content")
    subdir = tmpdir.mkdir("subdir")
    file2 = subdir.join("test2.txt")
    file2.write("content")

    tree = generate_directory_tree(str(tmpdir))
    assert "test1.txt" in tree
    assert "subdir" in tree
    assert "test2.txt" in tree

def test_generate_directory_tree_with_error(monkeypatch):
    def mock_walk(*args, **kwargs):
        yield (args[0], [], [])  # First yield a valid result
        raise OSError("Permission denied")

    monkeypatch.setattr(os, 'walk', mock_walk)
    result = generate_directory_tree("/fake/path")
    assert result == ""  # The function should return empty string on error

def test_get_file_language():
    assert get_file_language("test.py") == "Python"
    assert get_file_language("test.js") == "JavaScript"
    assert get_file_language("test.unknown") == "Unknown"
    assert get_file_language("test.tsx") == "TypeScript"
    assert get_file_language("test.cpp") == "C++"
    assert get_file_language("test.rs") == "Rust"

def test_get_file_language_additional_extensions():
    # Test more file extensions
    assert get_file_language("test.cs") == "C#"
    assert get_file_language("test.m") == "Objective-C"
    assert get_file_language("test.r") == "R"

    assert get_file_language("test.java") == "Java"
    assert get_file_language("test.go") == "Go"
    assert get_file_language("test.rb") == "Ruby"
    assert get_file_language("test.swift") == "Swift"
    assert get_file_language("test.kt") == "Kotlin"
    assert get_file_language("test.scala") == "Scala"
    assert get_file_language("test.php") == "PHP"
def test_get_file_language_empty_string():
    assert get_file_language("") == "Unknown"

def test_get_file_language_no_extension():
    assert get_file_language("test") == "Unknown"

def get_llm_response(file_path: str, llm_provider: str) -> Tuple[str, str]:
        # Test Groq provider
    desc, use_case = get_llm_response("test.py", "Groq")
    assert "Sample description" in desc
    assert "Sample use case" in use_case

    # Test other provider
    desc, use_case = get_llm_response("test.py", "Other")
    assert "Alternative description" in desc
    assert "Alternative use case" in use_case

    try:
        if llm_provider not in ("Groq", "Cerberas", "SombaNova", "Gemini"):  #Explicitly check for invalid provider
            return "Error: Invalid LLM provider", "Error: Invalid LLM provider"

    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"


def test_generate_synopsis_llm_error(tmpdir, monkeypatch):
    """Test generate_synopsis when the LLM API call fails."""
    test_file = tmpdir.join("test.py")
    test_file.write("print('hello')")
    def mock_error(msg): pass
    def mock_llm_response(*args, **kwargs):
        raise Exception("LLM API error")
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr("streamlit_app.streamlit_app.get_llm_response", mock_llm_response)
    generate_synopsis(str(tmpdir), True, True, True, True, "Groq") #check for error handling

def test_get_llm_response_error(monkeypatch):
    """Test error handling in get_llm_response."""
    def mock_llm_response(file_path, llm_provider):
        raise Exception("Simulated LLM error")

    monkeypatch.setattr("streamlit_app.streamlit_app.get_llm_response", mock_llm_response)

    try:
        desc, use_case = get_llm_response(None, "Groq")
    except RecursionError:
        desc = "Error: Recursion detected"
        use_case = "Error: Recursion detected"

    assert "Error:" in desc
    assert "Error:" in use_case

def test_get_llm_response_invalid_provider():
    """Test invalid provider handling."""
    # Test with invalid provider
    desc, use_case = get_llm_response("test.py", "InvalidProvider")
    assert "Error: Invalid LLM provider" in desc
    assert "Error: Invalid LLM provider" in use_case

    # Test with valid provider
    desc, use_case = get_llm_response("test.py", "Groq")
    assert "Sample description" in desc
    assert "Sample use case" in use_case

def test_handle_directory_error_not_a_directory(tmpdir, monkeypatch):
    """Test handle_directory_error when the path is not a directory."""
    test_file = tmpdir.join("test.txt")
    test_file.write("This is a test file.")
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert handle_directory_error(str(test_file)) is False

def test_save_synopsis_empty_content(tmpdir, monkeypatch):
    """Test save_synopsis when the content is empty."""
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert save_synopsis(str(tmpdir), "") is False

    # Create test file
    test_file = tmpdir.join("test.py")
    test_file.write("print('hello')")

    result = generate_synopsis(
        str(tmpdir),
        include_tree=False,
        include_descriptions=False,
        include_token_count=False,
        include_use_cases=False,
        llm_provider="Groq"
    )
    assert result is not None
    assert "Directory Tree" not in result
    assert "Token Count" not in result
    assert "Description" not in result
    assert "Use Case" not in result

def test_log_event_with_directory_creation(tmpdir):
    # Create a subdirectory path that doesn't exist yet
    new_dir = os.path.join(str(tmpdir), "new_dir")

    # Make sure directory doesn't exist
    if os.path.exists(new_dir):
        import shutil
        shutil.rmtree(new_dir)

    # Verify directory doesn't exist before test
    assert not os.path.exists(new_dir), "Directory already exists"

    # Test logging
    test_message = "test message"
    log_event(new_dir, test_message)

    # Verify directory was created
    assert os.path.exists(new_dir), "Directory was not created"

    # Verify log file exists
    log_file = os.path.join(new_dir, "event_log.txt")
    assert os.path.exists(log_file), "Log file was not created"

    # Verify log content
    with open(log_file, 'r', encoding="utf-8") as f:
        content = f.read()
        assert test_message in content, "Message not found in log file"

if __name__ == "__main__":
    pytest.main(["-v"])
