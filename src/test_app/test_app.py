"""This is the testing file for repo_synopsis"""

import os
import sys
from unittest.mock import mock_open
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app.streamlit_app import (
    handle_directory_error,
    save_synopsis,
    generate_synopsis,
    log_event,
    main,
    traverse_directory,
    generate_directory_tree,
    get_file_language,
    get_llm_response,
)


def mock_error(_msg):
    """Mock function to simulate error logging."""


def test_handle_directory_error_empty_path(monkeypatch):
    """Test handle_directory_error with an empty path."""
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error("") is False


def test_handle_directory_error_valid_path(tmpdir, monkeypatch):
    """Test handle_directory_error with a valid path."""
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error(str(tmpdir)) is True


def test_handle_directory_error_nonexistent_path(monkeypatch):
    """Test handle_directory_error with a nonexistent path."""
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error("/nonexistent/path") is False


def test_generate_synopsis_file_encoding_error(tmpdir, monkeypatch):
    """Test generate_synopsis with a file that causes an encoding error."""
    test_file = tmpdir.join("test.txt")
    test_file.write_binary(
        "This is a test file with invalid encoding\000".encode(
            "latin-1"
        )
    )
    monkeypatch.setattr(st, "error", mock_error)
    result = generate_synopsis(
        str(tmpdir),
        True, True, True, True, "Groq"
    )
    assert (
        result is None
        or "Unable to decode" in result
        or "Languages used: Unknown" in result
    )


def test_handle_directory_error_os_error(tmpdir, monkeypatch):
    """Test handle_directory_error with an OSError."""
    def mock_listdir(_path):
        raise OSError("Simulated OSError")
    monkeypatch.setattr(os, "listdir", mock_listdir)
    assert handle_directory_error(str(tmpdir)) is False


def test_handle_directory_error_permission_error(tmpdir, monkeypatch):
    """Test handle_directory_error with a permission error."""
    def mock_listdir(_path):
        raise PermissionError("Permission denied")
    monkeypatch.setattr(os, "listdir", mock_listdir)
    assert handle_directory_error(str(tmpdir)) is False


def test_save_synopsis_success(tmpdir, monkeypatch):
    """Test save_synopsis with a successful save."""
    def mock_success(_msg):
        pass

    m = mock_open()
    monkeypatch.setattr(st, "success", mock_success)
    monkeypatch.setattr("builtins.open", m)
    assert save_synopsis(str(tmpdir), "test content") is True


def test_log_event_success(tmpdir):
    """Test log_event with a successful log."""
    log_event(str(tmpdir), "test message")
    log_file = os.path.join(str(tmpdir), "event_log.txt")
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "test message" in content


def test_log_event_failure(tmpdir, monkeypatch):
    """Test log_event with a failed log due to IOError."""
    def mock_error_log(_msg):
        pass

    mock = mock_open()
    mock.side_effect = IOError("Simulated IO Error")
    monkeypatch.setattr(st, "error", mock_error_log)
    monkeypatch.setattr("builtins.open", mock)
    log_event(str(tmpdir), "test message")


def test_generate_synopsis_invalid_directory(monkeypatch):
    monkeypatch.setattr(st, "error", mock_error)
    assert generate_synopsis("", True, True, True, True, "Groq") is None


def test_generate_synopsis_empty_directory(tmpdir, monkeypatch):
    monkeypatch.setattr(st, "error", mock_error)
    assert generate_synopsis(
        str(tmpdir),
        True,
        True,
        True,
        True,
        "Groq"
    ) is None


def test_generate_synopsis_with_file_read_error(tmpdir, monkeypatch):
    """Test generate_synopsis with a file that causes a read error."""
    test_file = tmpdir.join("test.py")
    test_file.write("print('hello')")
    original_open = open  # Store the original open function

    def selective_mock_open(*args, **kwargs):
        """Mock function to simulate file read errors."""
        if args[0] == str(test_file):
            raise IOError("Simulated IO Error")
        return original_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", selective_mock_open)
    monkeypatch.setattr(st, "error", mock_error)
    result = generate_synopsis(str(tmpdir), True, True, True, True, "Groq")
    assert result is None or "Unable to read file" in result or "Languages used: Python" in result


def test_traverse_directory(tmpdir):
    file1 = tmpdir.join("test1.txt")
    file1.write("content")
    subdir = tmpdir.mkdir("subdir")
    file2 = subdir.join("test2.txt")
    file2.write("content")

    items = traverse_directory(str(tmpdir))
    assert len(items) == 2
    assert any("test1.txt" in item for item in items)


def test_traverse_directory_empty(tmpdir):
    """Test traverse_directory with an empty directory."""
    # Create an empty directory
    test_dir = tmpdir.mkdir("test_dir")

    # Call the function and check the result
    result = traverse_directory(str(test_dir))
    assert result == []


def test_traverse_directory_error(monkeypatch):
    def mock_walk(*_args, **_kwargs):
        raise OSError("Simulated OSError")

    monkeypatch.setattr(os, "walk", mock_walk)
    items = traverse_directory("/fake/path")
    assert items == []


def test_generate_directory_tree(tmpdir):
    """Test generate_directory_tree with a directory containing files and subdirectories."""
    file1 = tmpdir.join("test1.txt")
    file1.write("content")
    subdir = tmpdir.mkdir("subdir")
    file2 = subdir.join("test2.txt")
    file2.write("content")

    tree = generate_directory_tree(str(tmpdir))
    assert "test1.txt" in tree
    assert "subdir" in tree


def test_generate_directory_tree_with_error(monkeypatch):
    def mock_walk(*_args, **_kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "walk", mock_walk)
    tree = generate_directory_tree("/fake/path")
    assert tree == ""


def test_get_file_language():
    """Test get_file_language with various file extensions."""
    assert get_file_language("test.py") == "Python"
    assert get_file_language("test.unknown") == "Unknown"
    assert get_file_language("test.cs") == "C#"
    assert get_file_language("test.m") == "Objective-C"
    assert get_file_language("test.java") == "Java"
    assert get_file_language("test.go") == "Go"


def test_generate_synopsis_llm_error(tmpdir, monkeypatch):
    """Test generate_synopsis when the LLM API call fails."""
    def mock_success(_msg):
        """Mock function to simulate success logging."""
        pass

    def mock_llm_response(*_args, **_kwargs):
        """Mock function to simulate LLM API response failure."""
        raise ConnectionError("LLM API connection failed")

    def mock_generate_directory_tree(_path):
        """Mock function to simulate directory tree generation."""
        return "test.py"

    test_file = tmpdir.join("test.py")
    test_file.write("print('hello')")
    monkeypatch.setattr(st, "error", mock_error)
    monkeypatch.setattr(st, "success", mock_success)
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.get_llm_response",
        mock_llm_response
    )
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.generate_directory_tree",
        mock_generate_directory_tree
    )
    result = generate_synopsis(str(tmpdir), True, True, True, True, "Groq")
    assert result is None


def test_get_llm_response_invalid_provider():
    """Test get_llm_response with an invalid provider."""
    desc, use_case = get_llm_response("test.py", "InvalidProvider")
    assert desc == "Error: Invalid LLM provider"
    assert use_case == "Error: Invalid LLM provider"


def test_save_synopsis_empty_content(tmpdir, monkeypatch):
    """Test save_synopsis when the content is empty."""
    monkeypatch.setattr(st, "error", mock_error)
    result = save_synopsis(str(tmpdir), "")
    assert result is False


def test_handle_directory_error_not_a_directory(tmpdir, monkeypatch):
    """Test handle_directory_error when the path is not a directory."""
    test_file = tmpdir.join("test_file.txt")
    test_file.write("content")
    monkeypatch.setattr(st, "error", mock_error)
    result = handle_directory_error(str(test_file))
    assert result is False


def test_generate_synopsis_with_multiple_files(tmp_path, monkeypatch):
    """Test generate_synopsis with multiple files."""
    test_dir = tmp_path / "test_repo"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("print('hello')")
    (test_dir / "file2.py").write_text("class Test: pass")
    monkeypatch.setattr(st, "error", mock_error)
    result = generate_synopsis(str(test_dir), True, True, True, True, "Groq")
    assert result is not None


def test_traverse_directory_with_nested_structure(tmp_path):
    """Test traverse_directory with a nested directory structure."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "file1.py").write_text("content1")
    sub_dir = root / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file2.py").write_text("content2")

    files = traverse_directory(str(root))
    assert len(files) == 2
    assert any("file1.py" in f for f in files)
    assert any("file2.py" in f for f in files)


def test_generate_directory_tree_with_nested_structure(tmp_path):
    """Test generate_directory_tree with a nested directory structure."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "file1.py").write_text("content1")
    sub_dir = root / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file2.py").write_text("content2")

    tree = generate_directory_tree(str(root))
    assert "root" in tree
    assert "subdir" in tree
    assert "file1.py" in tree
    assert "file2.py" in tree


def test_main_with_error_handling(monkeypatch):
    """Test main function with error handling."""
    def mock_title(_msg):
        pass

    def mock_subheader(_msg):
        pass

    def mock_checkbox(_msg):
        return True

    def mock_selectbox(_msg, options):
        return options[0]

    def mock_text_input(_msg):
        return "/test/path"

    def mock_button(_msg):
        return True

    def mock_error_main(_msg):
        pass

    def mock_success(_msg):
        pass

    monkeypatch.setattr(st, "title", mock_title)
    monkeypatch.setattr(st, "subheader", mock_subheader)
    monkeypatch.setattr(st, "checkbox", mock_checkbox)
    monkeypatch.setattr(st, "selectbox", mock_selectbox)
    monkeypatch.setattr(st, "text_input", mock_text_input)
    monkeypatch.setattr(st, "button", mock_button)
    monkeypatch.setattr(st, "error", mock_error_main)
    monkeypatch.setattr(st, "success", mock_success)
    
    # Mock the st.checkbox function to avoid the TypeError
    def mock_checkbox_with_value(_label, value=False):
        return value 
    monkeypatch.setattr(st, "checkbox", mock_checkbox_with_value)

    main()

