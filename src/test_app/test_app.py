import pytest
import os
import streamlit as st
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

def test_handle_directory_error_nonexistent_path(monkeypatch):
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert handle_directory_error("/nonexistent/path") is False

def test_handle_directory_error_valid_path(tmpdir, monkeypatch):
    def mock_error(msg): pass
    monkeypatch.setattr(st, 'error', mock_error)
    assert handle_directory_error(str(tmpdir)) is True

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
    def mock_open(*args, **kwargs): raise IOError
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr("builtins.open", mock_open)
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
    def mock_open(*args, **kwargs): raise IOError
    monkeypatch.setattr(st, 'error', mock_error)
    monkeypatch.setattr("builtins.open", mock_open)
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

def test_get_file_language():
    assert get_file_language("test.py") == "Python"
    assert get_file_language("test.js") == "JavaScript"
    assert get_file_language("test.unknown") == "Unknown"
    assert get_file_language("test.tsx") == "TypeScript"
    assert get_file_language("test.cpp") == "C++"
    assert get_file_language("test.rs") == "Rust"

def test_get_llm_response():
    # Test Groq provider
    desc, use_case = get_llm_response("test.py", "Groq")
    assert "Sample description" in desc
    assert "Sample use case" in use_case

    # Test other provider
    desc, use_case = get_llm_response("test.py", "Other")
    assert "Alternative description" in desc
    assert "Alternative use case" in use_case

def test_get_llm_response_error(monkeypatch):
    def mock_raise(*args, **kwargs):
        raise Exception("API Error")
    monkeypatch.setattr("streamlit_app.streamlit_app.get_llm_response", mock_raise)

    desc, use_case = get_llm_response("test.py", "Groq")
    assert "Error: API Error" in desc
    assert "Error: API Error" in use_case

if __name__ == "__main__":
    pytest.main(["-v"])
