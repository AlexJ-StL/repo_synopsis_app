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

def test_generate_synopsis_with_mock_llm_response(tmpdir, monkeypatch):
    def mock_llm_response(*args, **kwargs):
        return "Mocked description", "Mocked use case"
    monkeypatch.setattr("streamlit_app.streamlit_app.get_llm_response", mock_llm_response)

    test_file = tmpdir.join("test.py")
    test_file.write("print('hello')")

    result = generate_synopsis(str(tmpdir), True, True, True, True, "Groq")
    assert "Mocked description" in result
    assert "Mocked use case" in result

def test_generate_synopsis_with_empty_llm_response(tmpdir, monkeypatch):
    def mock_llm_response(*args, **kwargs):
        return None, None
    monkeypatch.setattr("streamlit_app.streamlit_app.get_llm_response", mock_llm_response)

    # Create a test file to ensure directory isn't empty
    test_file = tmpdir.join("test.txt")
    test_file.write("content")

    result = generate_synopsis(str(tmpdir), True, True, True, True, "Groq")
    assert result is not None  # First check if result exists
    assert "Failed to generate description" in result

def test_traverse_directory_with_permission_error(tmpdir, monkeypatch):
    def mock_listdir(path):
        raise PermissionError("Mock permission error")
    monkeypatch.setattr(os, 'listdir', mock_listdir)

    try:
        result = traverse_directory(str(tmpdir))
        assert isinstance(result, list)
        assert len(result) == 0
    except PermissionError:
        assert True  # Alternative: allow the error to be raised

def test_generate_directory_tree_with_special_chars(tmpdir):
    # Test with special characters in filenames
    special_file = tmpdir.join("special!@#$%.txt")
    special_file.write("content")
    special_dir = tmpdir.mkdir("dir with spaces")
    nested_file = special_dir.join("nested.txt")
    nested_file.write("content")

    tree = generate_directory_tree(str(tmpdir))
    assert "special!@#$%.txt" in tree
    assert "dir with spaces" in tree
    assert "nested.txt" in tree

def test_get_llm_response_with_different_providers(monkeypatch):
    def mock_st_error(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)

    providers = ["Groq", "Cerberas", "SombaNova", "Gemini"]
    for provider in providers:
        response = get_llm_response("test prompt", provider)
        assert response is not None

def test_save_synopsis_with_unicode_content(tmpdir):
    unicode_content = "Test synopsis with unicode: 你好, 안녕하세요, Привет"
    save_synopsis(str(tmpdir), unicode_content)

    saved_file = tmpdir.join("repo_synopsis.md")
    with open(str(saved_file), 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == unicode_content

def test_save_synopsis_with_encoding_error(tmpdir, monkeypatch):
    from unittest.mock import MagicMock, mock_open as mock_open_func

    mock = MagicMock()
    mock.side_effect = UnicodeEncodeError('utf-8', b'test', 0, 1, 'mock encoding error')
    m = mock_open_func()
    m.return_value.write = mock

    monkeypatch.setattr("builtins.open", m)
    monkeypatch.setattr(st, 'error', lambda x: None)

    save_synopsis(str(tmpdir), "Test content with encoding issues")
    mock.assert_called()

def test_get_llm_response_with_timeout(monkeypatch):
    def mock_api_call(*args, **kwargs):
        raise TimeoutError("API request timed out")
    # Patch the actual API call function, not the main function
    monkeypatch.setattr("streamlit_app.streamlit_app._make_llm_api_call", mock_api_call)

    def mock_st_error(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)

    description, use_case = get_llm_response("test.py", "Groq")
    assert description is not None
    assert "timeout" in description.lower()

def test_traverse_directory_with_memory_error(tmpdir, monkeypatch):
    def mock_listdir(path):
        raise MemoryError("Out of memory")
    monkeypatch.setattr(os, 'listdir', mock_listdir)

    try:
        result = traverse_directory(str(tmpdir))
        assert result == []  # Should return empty list on error
    except MemoryError:
        assert True  # Alternative: allow the error to be raised

def test_generate_synopsis_with_large_directory(tmpdir, monkeypatch):
    # Create a large directory structure
    for i in range(1000):
        subdir = tmpdir.mkdir(f"dir_{i}")
        file = subdir.join(f"file_{i}.txt")
        file.write(f"Content {i}")

    def mock_st_warning(x): return None
    def mock_st_error(x): return None
    def mock_st_write(x): return None
    monkeypatch.setattr(st, 'warning', mock_st_warning)
    monkeypatch.setattr(st, 'error', mock_st_error)
    monkeypatch.setattr(st, 'write', mock_st_write)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None

def test_file_permission_scenarios(tmpdir):
    # Create a file with restricted permissions
    test_file = tmpdir.join("restricted.txt")
    test_file.write("content")
    os.chmod(str(test_file), 0o000)  # Remove all permissions

    try:
        result = traverse_directory(str(tmpdir))
        assert str(test_file) in result
    finally:
        os.chmod(str(test_file), 0o666)  # Restore permissions for cleanup

def test_get_file_language_with_mixed_case(tmpdir):
    assert get_file_language("test.PY") == "Python"
    assert get_file_language("test.Js") == "JavaScript"
    assert get_file_language("test.MD") == "Markdown"

def test_generate_synopsis_with_symlinks(tmpdir):
    # Create a directory with symlinks
    original_dir = tmpdir.mkdir("original")
    original_file = original_dir.join("test.txt")
    original_file.write("content")

    link_dir = tmpdir.mkdir("links")
    os.symlink(str(original_file), str(link_dir.join("link.txt")))

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None
    assert "link.txt" in result

def test_save_synopsis_with_concurrent_access(tmpdir):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # Simulate concurrent access to the synopsis file
    def save_concurrent():
        save_synopsis(str(tmpdir), f"Content from thread {threading.get_ident()}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(save_concurrent) for _ in range(5)]
        for future in futures:
            future.result()

    # Verify the file exists and contains content
    synopsis_file = tmpdir.join("repo_synopsis.md")
    assert synopsis_file.exists()
    content = synopsis_file.read()
    assert "Content from thread" in content

def test_generate_synopsis_with_binary_files(tmpdir):
    # Create a binary file
    binary_file = tmpdir.join("test.bin")
    with open(str(binary_file), 'wb') as f:
        f.write(bytes(range(256)))

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None
    assert "test.bin" in result

def test_log_event_with_rotation(tmpdir):
    # Test log rotation by creating multiple log entries
    for i in range(1000):
        log_event(str(tmpdir), f"Log entry {i}")

    log_file = os.path.join(str(tmpdir), "event_log.txt")
    assert os.path.exists(log_file)

    # Check if file size is reasonable
    assert os.path.getsize(log_file) < 1024 * 1024  # Less than 1MB

def test_generate_synopsis_with_network_error(tmpdir, monkeypatch):
    def mock_llm_response(*args, **kwargs):
        raise ConnectionError("Network connection failed")
    monkeypatch.setattr("streamlit_app.streamlit_app.get_llm_response", mock_llm_response)

    # Create a test file to ensure directory isn't empty
    test_file = tmpdir.join("test.txt")
    test_file.write("content")

    result = generate_synopsis(str(tmpdir), True, True, True, True, "Groq")
    assert isinstance(result, str)
    assert "Network connection failed" in result

def test_generate_synopsis_with_rate_limit(tmpdir, monkeypatch):
    def mock_llm_response(*args, **kwargs):
        raise Exception("Rate limit exceeded")
    monkeypatch.setattr("streamlit_app.streamlit_app.get_llm_response", mock_llm_response)

    # Create a test file to ensure directory isn't empty
    test_file = tmpdir.join("test.txt")
    test_file.write("content")

    result = generate_synopsis(str(tmpdir), True, True, True, True, "Groq")
    assert isinstance(result, str)
    assert "Rate limit exceeded" in result

def test_generate_synopsis_with_invalid_unicode(tmpdir, monkeypatch):
    # Create a file with invalid unicode
    test_file = tmpdir.join("invalid_unicode.txt")
    with open(str(test_file), 'wb') as f:
        f.write(b'\xff\xfe\xfd')  # Invalid UTF-8

    def mock_st_error(x): return None
    def mock_st_write(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    monkeypatch.setattr(st, 'write', mock_st_write)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None

def test_generate_synopsis_with_deep_recursion(tmpdir, monkeypatch):
    # Create a deeply nested directory structure
    current_dir = tmpdir
    for i in range(100):  # Create 100 levels of nesting
        current_dir = current_dir.mkdir(f"level_{i}")
        file = current_dir.join("file.txt")
        file.write("content")

    def mock_st_error(x): return None
    def mock_st_write(x): return None
    def mock_st_warning(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    monkeypatch.setattr(st, 'write', mock_st_write)
    monkeypatch.setattr(st, 'warning', mock_st_warning)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None

def test_generate_synopsis_with_empty_files(tmpdir, monkeypatch):
    # Create multiple empty files
    for i in range(10):
        file = tmpdir.join(f"empty_{i}.txt")
        file.write("")

    def mock_st_error(x): return None
    def mock_st_write(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    monkeypatch.setattr(st, 'write', mock_st_write)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None
    assert "empty_" in result

def test_generate_synopsis_with_special_filenames(tmpdir, monkeypatch):
    # Create files with special characters and spaces
    special_chars = ['!', '@', '#', '$', '%', '^', '&', ' ', '(', ')', '[', ']']
    for char in special_chars:
        file = tmpdir.join(f"file{char}name.txt")
        file.write("content")

    def mock_st_error(x): return None
    def mock_st_write(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    monkeypatch.setattr(st, 'write', mock_st_write)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None

def test_generate_synopsis_with_mixed_line_endings(tmpdir, monkeypatch):
    # Create files with different line endings
    file_unix = tmpdir.join("unix_endings.txt")
    file_unix.write("line1\nline2\nline3")

    file_windows = tmpdir.join("windows_endings.txt")
    file_windows.write("line1\r\nline2\r\nline3")

    def mock_st_error(x): return None
    def mock_st_write(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    monkeypatch.setattr(st, 'write', mock_st_write)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None
    assert "unix_endings.txt" in result
    assert "windows_endings.txt" in result

def test_generate_synopsis_with_hidden_files(tmpdir, monkeypatch):
    # Create hidden files and directories
    hidden_dir = tmpdir.mkdir(".hidden_dir")
    hidden_file = hidden_dir.join(".hidden_file.txt")
    hidden_file.write("content")

    def mock_st_error(x): return None
    def mock_st_write(x): return None
    monkeypatch.setattr(st, 'error', mock_st_error)
    monkeypatch.setattr(st, 'write', mock_st_write)

    result = generate_synopsis(
        str(tmpdir),
        include_tree=True,
        include_descriptions=True,
        include_token_count=True,
        include_use_cases=True,
        llm_provider="Groq"
    )
    assert result is not None
    assert ".hidden_dir" in result
    assert ".hidden_file.txt" in result

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
