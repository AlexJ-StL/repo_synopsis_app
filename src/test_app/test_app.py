import pytest
import os
import streamlit as st
from streamlit_app.streamlit_app import (
    handle_directory_error,
    save_synopsis,
    generate_synopsis,
    log_event,
    main
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

if __name__ == "__main__":
    pytest.main(["-v"])
