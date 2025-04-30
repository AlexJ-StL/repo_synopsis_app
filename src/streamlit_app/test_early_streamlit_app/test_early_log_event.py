# test_streamlit_app_log_event.py

import os
import tempfile
import shutil
import io
import sys
import pytest

from streamlit_app.streamlit_app import log_event

@pytest.mark.usefixtures("clean_log_env")
class TestLogEvent:
    """Unit tests for the log_event function in streamlit_app.streamlit_app."""

    # --- Happy Path Tests ---

    @pytest.mark.happy_path
    def test_log_event_creates_log_file_and_writes_message(self, tmp_path):
        """
        Test that log_event creates 'event_log.txt' in the given directory and writes the message.
        """
        test_dir = tmp_path / "logs"
        message = "Test log message"
        log_event(str(test_dir), message)
        log_file = test_dir / "event_log.txt"
        assert log_file.exists(), "Log file should be created"
        content = log_file.read_text(encoding="utf-8")
        assert message in content, "Log message should be present in the log file"
        assert len(content.splitlines()) == 1, "Only one log entry should be present"

    @pytest.mark.happy_path
    def test_log_event_appends_multiple_messages(self, tmp_path):
        """
        Test that log_event appends messages to the same log file.
        """
        test_dir = tmp_path
        messages = ["First message", "Second message", "Third message"]
        for msg in messages:
            log_event(str(test_dir), msg)
        log_file = test_dir / "event_log.txt"
        content = log_file.read_text(encoding="utf-8")
        for msg in messages:
            assert msg in content, f"Message '{msg}' should be in the log file"
        assert len(content.splitlines()) == len(messages), "Each message should be a new line"

    @pytest.mark.happy_path
    def test_log_event_uses_current_directory_when_path_is_empty(self, tmp_path, monkeypatch):
        """
        Test that log_event logs to the current directory if directory_path is empty.
        """
        # Change working directory to a temp dir
        monkeypatch.chdir(tmp_path)
        message = "Logging to current directory"
        log_event("", message)
        log_file = tmp_path / "event_log.txt"
        assert log_file.exists(), "Log file should be created in current directory"
        content = log_file.read_text(encoding="utf-8")
        assert message in content

    @pytest.mark.happy_path
    def test_log_event_uses_current_directory_when_path_is_none(self, tmp_path, monkeypatch):
        """
        Test that log_event logs to the current directory if directory_path is None.
        """
        monkeypatch.chdir(tmp_path)
        message = "Logging with None path"
        log_event(None, message)
        log_file = tmp_path / "event_log.txt"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert message in content

    # --- Edge Case Tests ---

    @pytest.mark.edge_case
    def test_log_event_handles_nonexistent_nested_directories(self, tmp_path):
        """
        Test that log_event creates nested directories if they do not exist.
        """
        nested_dir = tmp_path / "a" / "b" / "c"
        message = "Nested directory log"
        log_event(str(nested_dir), message)
        log_file = nested_dir / "event_log.txt"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert message in content

    @pytest.mark.edge_case
    def test_log_event_handles_directory_path_as_dot(self, tmp_path, monkeypatch):
        """
        Test that log_event logs to the current directory when directory_path is '.'.
        """
        monkeypatch.chdir(tmp_path)
        message = "Dot directory log"
        log_event(".", message)
        log_file = tmp_path / "event_log.txt"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert message in content

    @pytest.mark.edge_case
    def test_log_event_handles_directory_path_as_whitespace(self, tmp_path, monkeypatch):
        """
        Test that log_event logs to the current directory when directory_path is whitespace.
        """
        monkeypatch.chdir(tmp_path)
        message = "Whitespace directory log"
        log_event("   ", message)
        log_file = tmp_path / "event_log.txt"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert message in content

    @pytest.mark.edge_case
    def test_log_event_handles_oserror_on_open(monkeypatch, tmp_path):
        """
        Test that log_event handles OSError when file cannot be opened for writing.
        """
        test_dir = tmp_path
        message = "Should not be written"
        log_file = test_dir / "event_log.txt"
        log_file.write_text("existing", encoding="utf-8")
        # Make the file read-only
        os.chmod(log_file, 0o444)
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        monkeypatch.setattr(sys, "stderr", captured)
        log_event(str(test_dir), message)
        os.chmod(log_file, 0o666)  # Restore permissions for cleanup
        output = captured.getvalue()
        assert "Error writing to log file" in output or "Permission denied" in output

    @pytest.mark.edge_case
    def test_log_event_handles_unexpected_exception(monkeypatch, tmp_path):
        """
        Test that log_event handles unexpected exceptions (simulate by patching open to raise).
        """
        test_dir = tmp_path
        message = "Should not be written"
        def raise_exception(*args, **kwargs):
            raise RuntimeError("Unexpected error!")
        monkeypatch.setattr("builtins.open", raise_exception)
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        monkeypatch.setattr(sys, "stderr", captured)
        log_event(str(test_dir), message)
        output = captured.getvalue()
        assert "Unexpected error during logging" in output
        assert "Unexpected error!" in output

    @pytest.mark.edge_case
    def test_log_event_handles_non_string_message(self, tmp_path):
        """
        Test that log_event can handle non-string messages (should coerce to string).
        """
        test_dir = tmp_path
        message = 12345  # int, not str
        log_event(str(test_dir), message)
        log_file = test_dir / "event_log.txt"
        content = log_file.read_text(encoding="utf-8")
        assert "12345" in content

    @pytest.mark.edge_case
    def test_log_event_handles_long_message(self, tmp_path):
        """
        Test that log_event can handle a very long message.
        """
        test_dir = tmp_path
        message = "A" * 10000
        log_event(str(test_dir), message)
        log_file = test_dir / "event_log.txt"
        content = log_file.read_text(encoding="utf-8")
        assert message in content

    @pytest.mark.edge_case
    def test_log_event_handles_special_characters_in_message(self, tmp_path):
        """
        Test that log_event can handle messages with special characters.
        """
        test_dir = tmp_path
        message = "Special chars: äöüß€\n\t\r"
        log_event(str(test_dir), message)
        log_file = test_dir / "event_log.txt"
        content = log_file.read_text(encoding="utf-8")
        assert "äöüß€" in content
        assert "\\n" not in content  # Should write literal newlines, not escaped

    @pytest.mark.edge_case
    def test_log_event_handles_directory_path_with_trailing_slash(self, tmp_path):
        """
        Test that log_event works with directory paths that have a trailing slash.
        """
        test_dir = tmp_path / "withslash"
        os.makedirs(test_dir, exist_ok=True)
        dir_with_slash = str(test_dir) + os.sep
        message = "Trailing slash"
        log_event(dir_with_slash, message)
        log_file = test_dir / "event_log.txt"
        content = log_file.read_text(encoding="utf-8")
        assert message in content

    @pytest.mark.edge_case
    def test_log_event_handles_directory_path_with_invalid_characters(self, tmp_path):
        """
        Test that log_event handles directory paths with invalid characters gracefully.
        """
        # On most OSes, "?" is invalid in directory names
        invalid_dir = str(tmp_path / "invalid?dir")
        message = "Invalid dir"
        captured = io.StringIO()
        sys_stdout = sys.stdout
        sys.stderr = captured
        sys.stdout = captured
        try:
            log_event(invalid_dir, message)
        finally:
            sys.stdout = sys_stdout
            sys.stderr = sys_stdout
        output = captured.getvalue()
        assert "Error writing to log file" in output or "Original log message" in output

# --- Fixtures ---

@pytest.fixture
def clean_log_env(tmp_path):
    """
    Fixture to ensure a clean environment for log files.
    """
    yield
    # Cleanup is handled by tmp_path fixture