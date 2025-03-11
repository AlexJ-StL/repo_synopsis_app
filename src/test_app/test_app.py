"""This is the testing file for repo_synopsis"""
import os
import sys
# Standard library imports
from unittest.mock import patch
# Third party imports
import streamlit as st
import pytest
# Local application imports
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
    process_repo,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mock_error(_msg):
    """Mock function to simulate error logging."""


def test_handle_directory_error_empty_path(monkeypatch):
    """Test handle_directory_error with an empty path."""
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error("") is False


def test_handle_directory_error_valid_path(tmp_path):
    """Test handle_directory_error with a valid path."""

    assert handle_directory_error(str(tmp_path)) is True


def test_handle_directory_error_nonexistent_path(monkeypatch):
    """Test handle_directory_error with a nonexistent path."""
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error("/nonexistent/path") is False


def test_handle_directory_error_os_error(tmp_path, monkeypatch):
    """Test handle_directory_error with an OSError."""
    with patch("os.listdir", side_effect=OSError("Simulated OSError")):
        monkeypatch.setattr(st, "error", mock_error)
        assert handle_directory_error(str(tmp_path)) is False


def test_handle_directory_error_permission_error(tmp_path, monkeypatch):
    """Test handle_directory_error with a permission error."""
    with patch("os.listdir", side_effect=PermissionError("Permission denied")):
        monkeypatch.setattr(st, "error", mock_error)
        assert handle_directory_error(str(tmp_path)) is False


def test_save_synopsis_success(tmp_path, monkeypatch):
    """Test save_synopsis with a successful save."""
    monkeypatch.setattr(st, "success", lambda msg: None)
    assert save_synopsis(str(tmp_path), "test content") is True


def test_log_event_success(tmp_path):
    """Test log_event with a successful log."""
    log_event(str(tmp_path), "test message")
    log_file = os.path.join(str(tmp_path), "event_log.txt")
    with open(log_file, "r", encoding="utf-8") as file:
        content = file.read()
        assert "test message" in content


def test_log_event_failure(tmp_path, monkeypatch):
    """Test log_event with a failed log due to IOError."""
    monkeypatch.setattr(st, "error", mock_error)
    with patch("builtins.open", side_effect=IOError("Simulated IO Error")):
        log_event(str(tmp_path), "test message")


def test_generate_synopsis_invalid_directory(monkeypatch):
    """Test generate_synopsis with an invalid directory."""
    monkeypatch.setattr(st, "error", mock_error)
    assert generate_synopsis("", True, True, True, True, "Groq") is None


def test_generate_synopsis_empty_directory(tmp_path, monkeypatch):
    """Test generate_synopsis with an empty directory."""
    monkeypatch.setattr(st, "error", mock_error)
    assert generate_synopsis(
        str(tmp_path),
        True,
        True,
        True,
        True,
        "Groq"
    ) is None


def test_traverse_directory(tmp_path):
    """Test traverse_directory with files and subdirectories."""
    file1 = tmp_path / "test1.txt"
    file1.write_text("content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file2 = subdir / "test2.txt"
    file2.write_text("content")

    items = traverse_directory(str(tmp_path))
    assert len(items) == 2
    assert any("test1.txt" in item for item in items)
    assert any("test2.txt" in item for item in items)


def test_traverse_directory_empty(tmp_path):
    """Test traverse_directory with an empty directory."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    result = traverse_directory(str(empty_dir))
    assert not result


def test_traverse_directory_error():
    """Test traverse_directory with an OSError."""
    with patch("os.walk", side_effect=OSError("Simulated OSError")):
        items = traverse_directory("/fake/path")
        assert not items


def test_generate_directory_tree(tmp_path):
    """Test generate_directory_tree with a
    directory containing files and subdirectories."""
    file1 = tmp_path / "test1.txt"
    file1.write_text("content")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file2 = subdir / "test2.txt"
    file2.write_text("content")

    tree = generate_directory_tree(str(tmp_path))
    assert "test1.txt" in tree
    assert "subdir" in tree
    assert "test2.txt" in tree


def test_generate_directory_tree_with_error():
    """Test generate_directory_tree with an OSError."""
    with patch("os.walk", side_effect=OSError("Permission denied")):
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


@patch("streamlit_app.streamlit_app.summarize_text")
def test_get_llm_response(mock_summarize, tmp_path):
    """Test get_llm_response."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text content")
    mock_summarize.return_value = "Summarized text"
    description, use_case = get_llm_response(str(file_path), "OpenAI")
    assert description == "Summarized text"
    assert use_case == "Placeholder use case"


def test_get_llm_response_error(tmp_path):
    """Test get_llm_response with an IOError."""
    file_path = tmp_path / "test.txt"
    with patch("builtins.open", side_effect=IOError("Simulated IO Error")):
        description, use_case = get_llm_response(str(file_path), "OpenAI")
        assert "Error" in description
        assert "Error" in use_case


def test_process_repo_success(tmp_path, monkeypatch):
    """Test process_repo successfully."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "file1.py").write_text("print('hello')")
    include_options = {
        "token_count": True,
        "descriptions": True,
        "use_cases": True
    }
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.get_llm_response",
        lambda *args: ("desc", "use")
    )
    repo_data = process_repo(str(repo_path), include_options, "OpenAI")
    assert "files" in repo_data
    assert len(repo_data["files"]) == 1
    assert repo_data["files"][0]["language"] == "Python"


def test_process_repo_error(tmp_path, monkeypatch):
    """Test process_repo with an error."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "file1.py").write_text("print('hello')")
    include_options = {
        "token_count": True,
        "descriptions": True,
        "use_cases": True
    }
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.get_llm_response",
        lambda *args: ("desc", "use")
    )
    monkeypatch.setattr(st, "error", mock_error)
    with patch("os.walk", side_effect=OSError("Simulated OSError")):
        repo_data = process_repo(str(repo_path), include_options, "OpenAI")
        assert "error" in repo_data


def test_save_synopsis_empty_content(tmp_path, monkeypatch):
    """Test save_synopsis when the content is empty."""
    monkeypatch.setattr(st, "error", mock_error)
    result = save_synopsis(str(tmp_path), "")
    assert result is False


def test_handle_directory_error_not_a_directory(tmp_path, monkeypatch):
    """Test handle_directory_error when the path is not a directory."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("content")
    monkeypatch.setattr(st, "error", mock_error)
    result = handle_directory_error(str(test_file))
    assert result is False


@pytest.mark.parametrize("include_option, expected_result", [
    (
        {"tree": True,
            "descriptions": False,
            "token_count": False,
            "use_cases": False}, True
    ),
    (
        {"tree": False,
            "descriptions": True,
            "token_count": True,
            "use_cases": True}, True
    ),
    (
        {"tree": False,
            "descriptions": False,
            "token_count": False,
            "use_cases": False}, True
    ),
])
def test_generate_synopsis_various_options(
    tmp_path,
    include_option,
    expected_result,
    monkeypatch
):
    """Test generate_synopsis with various options."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    monkeypatch.setattr(st, "error", mock_error)
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.get_llm_response",
        lambda *args: ("desc", "use")
    )
    result = generate_synopsis(
        str(tmp_path),
        **include_option,
        llm_provider="Groq"
    )
    assert (result is not None) == expected_result


def test_generate_synopsis_with_multiple_files(tmp_path, monkeypatch):
    """Test generate_synopsis with multiple files."""
    test_dir = tmp_path / "test_repo"
    test_dir.mkdir()
    (test_dir / "file1.py").write_text("print('hello')")
    (test_dir / "file2.py").write_text("class Test: pass")
    monkeypatch.setattr(st, "error", mock_error)
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.get_llm_response",
        lambda *args: ("desc", "use")
    )
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


@patch(
    "streamlit_app.streamlit_app.st.button",
    return_value=True
)
@patch(
    "streamlit_app.streamlit_app.st.text_input",
    return_value="test_dir"
)
@patch(
    "streamlit_app.streamlit_app.st.multiselect",
    return_value=["subdir"]
)
@patch(
    "streamlit_app.streamlit_app.st.selectbox",
    return_value="Gemnini"
)
@patch(
    "streamlit_app.streamlit_app.st.checkbox",
    side_effect=[True, True, True, True]
)
@patch(
    "streamlit_app.streamlit_app.os.listdir",
    return_value=["subdir"]
)
@patch(
    "streamlit_app.streamlit_app.os.path.isdir",
    return_value=True
)
@patch(
    "streamlit_app.streamlit_app.process_repo",
    return_value={"repo_path": "/test/path/test_repo", "files": []}
)
@patch("streamlit_app.streamlit_app.json.dump")
@patch("streamlit_app.streamlit_app.st.success")
@patch("streamlit_app.streamlit_app.st.warning")
@patch("streamlit_app.streamlit_app.st.error")
def test_main_success(
    mocked_st_error,
    mocked_st_warning,
    mock_success,
    mock_json_dump,
    mock_process_repo,
    _mock_isdir,
    _mock_listdir,
    _mock_checkbox,
    _mock_selectbox,
    _mock_multiselect,
    _mock_text_input,
    _mock_button,
):
    """Test the main function successfully."""
    mocked_st_error.side_effect = lambda msg: None
    # Suppress Streamlit errors during testing
    mocked_st_warning.side_effect = lambda msg: None
    # Suppress Streamlit warnings
    mock_success.side_effect = lambda msg: None
    # Suppress Streamlit success messages
    mock_json_dump.side_effect = lambda data, f, indent: None
    # Suppress json.dump

    # Crucial change: Making sure the function being tested actually runs
    main()

    mock_process_repo.assert_called_once()
    mocked_st_error.assert_not_called()
    mocked_st_warning.assert_not_called()
