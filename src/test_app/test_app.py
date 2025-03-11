"""This is the testing file for repo_synopsis"""
import os
import sys
import shutil
# Standard library imports
from unittest.mock import patch
from pathlib import Path
from typing import NamedTuple

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
    summarize_text,
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
    result = generate_synopsis(
        str(tmp_path),
        True,
        True,
        True,
        True,
        "Groq"
    )
    assert result == ""  # Assert that an empty string is returned


def test_generate_synopsis_no_items(tmp_path, monkeypatch):
    """Test generate_synopsis when no items are found."""
    monkeypatch.setattr(st, "error", mock_error)  # Mock Streamlit warnings
    result = generate_synopsis(
        str(tmp_path),
        True,
        False,
        False,
        False,
        "Groq"
    )
    assert result == ""  # Assert that an empty string is returned


def test_generate_synopsis_all_options_false(tmp_path, monkeypatch):
    """Test generate_synopsis with all options set to False."""
    monkeypatch.setattr(st, "error", mock_error)
    result = generate_synopsis(
        str(tmp_path),
        False,
        False,
        False,
        False,
        "Groq"
    )
    assert result is not None  # Should still produce an empty synopsis


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


def test_summarize_text():
    """Test the text summarization function."""
    text = """This is a long piece of text that
    needs to be summarized. It contains multiple
    sentences and should be made shorter by the
    summarization function. The summary should capture
    the main points while reducing the overall length."""
    summary = summarize_text(text)
    assert len(summary) < len(text)  # Check for length reduction
    assert "summarized" in summary.lower()  # Check for content


def test_get_file_language_known():
    """Test get_file_language with known extensions."""
    assert get_file_language("my_file.py") == "Python"
    assert get_file_language("script.js") == "JavaScript"
    assert get_file_language("data.json") == "JSON"


def test_get_file_language_unknown():
    """Test get_file_language with unknown extensions."""
    assert get_file_language("mystery.xyz") == "Unknown"


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

    description, use_case = get_llm_response(str(file_path), "Groq")

    assert description == "Summarized text"
    assert use_case == "Placeholder use case"


def test_get_llm_response_error(tmp_path):
    """Test get_llm_response with an IOError."""
    file_path = tmp_path / "test.txt"
    with patch("builtins.open", side_effect=IOError("Simulated IO Error")):
        description, use_case = get_llm_response(str(file_path), "Groq")
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
    repo_data = process_repo(str(repo_path), include_options, "Groq")
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
        repo_data = process_repo(str(repo_path), include_options, "Groq")
        assert "error" not in repo_data


def test_process_repo_empty_repo(monkeypatch):
    """Test processing empty repositories"""
    monkeypatch.setattr(st, "error", mock_error)
    result = process_repo(
        "",
        {
            "token_count": False,
            "descriptions": False,
            "use_cases": False
        },
        "Groq"
    )
    assert "error" in result
    assert result["error"] == "Empty repo path provided"


def test_process_repo_no_files(tmp_path, monkeypatch):
    """Test when a directory is given with no files"""
    repo_path = tmp_path / "empty_repo"
    repo_path.mkdir()
    monkeypatch.setattr(st, "error", mock_error)
    result = process_repo(
        str(repo_path),
        {
            "token_count": False,
            "descriptions": False,
            "use_cases": False
        },
        "Groq"
    )
    assert "files" in result
    assert len(result["files"]) == 0


def test_process_repo_unknown_language(tmp_path, monkeypatch):
    """Test process_repo when an unknown file type is encountered."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / "file1.xyz").write_text("content")
    include_options = {
        "token_count": False,
        "descriptions": False,
        "use_cases": False
    }
    repo_data = process_repo(str(repo_path), include_options, "Groq")
    assert "files" in repo_data
    assert len(repo_data["files"]) == 1
    assert repo_data["files"][0]["language"] == "Unknown"


def test_process_repo_file_read_error(tmp_path, monkeypatch):
    """Test process_repo when file reading fails."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    file_path = repo_path / "file1.py"
    file_path.touch()  # Create file without content
    include_options = {
        "token_count": True,
        "descriptions": False,
        "use_cases": False
    }

    # Mock the built-in 'open' function to raise an IOError
    with patch('builtins.open', side_effect=IOError("Simulated IOError")):
        repo_data = process_repo(str(repo_path), include_options, "Groq")

    assert "files" in repo_data
    assert len(repo_data["files"]) == 1
    assert repo_data["files"][0]["token_count"] == "Unable to read file"


def test_save_synopsis_with_content(tmp_path):
    """Test saving synopsis with content."""
    synopsis_content = "This is the synopsis content."
    result = save_synopsis(str(tmp_path), synopsis_content)
    assert result is True
    synopsis_file = tmp_path / "synopsis.txt"
    assert synopsis_file.exists()
    assert synopsis_file.read_text() == synopsis_content


def test_save_synopsis_empty_content(tmp_path, monkeypatch):
    """Test save_synopsis when the content is empty."""
    monkeypatch.setattr(st, "error", mock_error)
    result = save_synopsis(str(tmp_path), "")
    assert result is False


def test_save_synopsis_ioerror(tmp_path, monkeypatch):
    """Test save_synopsis with an IOError."""
    monkeypatch.setattr(st, "error", mock_error)
    with patch("builtins.open", side_effect=IOError("Simulated IOError")):
        result = save_synopsis(str(tmp_path), "Test")
        assert result is False


def test_handle_directory_error_not_a_directory(tmp_path, monkeypatch):
    """Test handle_directory_error when the path is not a directory."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("content")
    monkeypatch.setattr(st, "error", mock_error)
    result = handle_directory_error(str(test_file))
    assert result is False


def test_handle_directory_error_invalid_path(monkeypatch):
    """Test invalid path scenarios"""
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error("nonexistent_path") is False
    assert handle_directory_error("/this/is/also/bad") is False
    # Test other cases


def test_handle_directory_error_file_path(tmp_path, monkeypatch):
    """Test file path scenarios"""
    file_path = tmp_path / "a_file.txt"
    file_path.touch()
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error(str(file_path)) is False


def test_handle_directory_error_no_permission(tmp_path, monkeypatch):
    """Test handle_directory_error when permission is denied."""
    test_dir = tmp_path / "testdir"
    test_dir.mkdir()

    # Mock os.listdir to raise PermissionError
    monkeypatch.setattr(
        "os.listdir",
        lambda _: (_ for _ in ()).throw(PermissionError("Permission denied"))
    )
    monkeypatch.setattr(st, "error", mock_error)

    assert handle_directory_error(str(test_dir)) is False


def test_handle_directory_error_file_exists(tmp_path, monkeypatch):
    """Test handle_directory_error when file exists."""
    test_file = tmp_path / "test_file.txt"
    test_file.touch()
    monkeypatch.setattr(st, "error", mock_error)
    assert handle_directory_error(str(test_file)) is False


class SynopsisTestCase(NamedTuple):
    """Test case structure for generate_synopsis tests."""
    include_tree: bool
    include_descriptions: bool
    include_token_count: bool
    include_use_cases: bool
    expected_result: bool


@pytest.mark.parametrize(
    "test_case",
    [
        SynopsisTestCase(
            include_tree=True,
            include_descriptions=False,
            include_token_count=False,
            include_use_cases=False,
            expected_result=True
        ),  # Tree only
        SynopsisTestCase(
            include_tree=False,
            include_descriptions=True,
            include_token_count=True,
            include_use_cases=True,
            expected_result=True
        ),  # All details, no tree
        SynopsisTestCase(
            include_tree=False,
            include_descriptions=False,
            include_token_count=False,
            include_use_cases=False,
            expected_result=False
        ),  # No content
    ]
)
def test_generate_synopsis_various_options(
    tmp_path,
    monkeypatch,
    test_case: SynopsisTestCase
):
    """Test generate_synopsis with various options."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    monkeypatch.setattr(st, "error", lambda msg: None)
    monkeypatch.setattr(
        "streamlit_app.streamlit_app.get_llm_response",
        lambda *args: ("desc", "use")
    )

    result = generate_synopsis(
        str(tmp_path),
        include_tree=test_case.include_tree,
        include_descriptions=test_case.include_descriptions,
        include_token_count=test_case.include_token_count,
        include_use_cases=test_case.include_use_cases,
        llm_provider="Groq"
    )

    assert (result is not None) == test_case.expected_result


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
    return_value="Groq"
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
    return_value={
        "repo_path": "/test/path/test_repo",
        "files": []
    }
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

    # Create a dummy directory for testing
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "subdir").mkdir(exist_ok=True)  # Create subdir
    (test_dir / "subdir").mkdir(exist_ok=True)

    # Crucial change: Making sure the function being tested actually runs
    main()

    mock_process_repo.assert_called_once()
    mocked_st_error.assert_not_called()
    mocked_st_warning.assert_not_called()

    shutil.rmtree("test_dir")
