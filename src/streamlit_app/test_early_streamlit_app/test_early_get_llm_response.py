# test_streamlit_app_get_llm_response.py

import os
import io
import pytest

from unittest import mock

# Import the function to test
from streamlit_app.streamlit_app import get_llm_response

@pytest.mark.usefixtures("tmp_path")
class TestGetLlmResponse:
    # --- HAPPY PATHS ---

    @pytest.mark.happy_path
    def test_small_file_core_logic(self, tmp_path):
        """Test: Small file, normal content, filename triggers 'Core Logic/Component' use case."""
        file_path = tmp_path / "main.py"
        content = "def foo():\n    '''This is a function.'''\n    pass\n" * 20  # >30 words
        file_path.write_text(content)
        # Patch summarize_text to return a known summary
        with mock.patch("streamlit_app.streamlit_app.summarize_text", return_value="Summary here"):
            desc, use_case = get_llm_response(str(file_path), "any")
        assert desc == "Summary here"
        assert use_case == "Core Logic/Component"

    @pytest.mark.happy_path
    def test_small_file_testing_use_case(self, tmp_path):
        """Test: File with 'test' in name triggers 'Testing/Verification' use case."""
        file_path = tmp_path / "test_utils.py"
        content = "def test_foo():\n    assert True\n" * 20
        file_path.write_text(content)
        with mock.patch("streamlit_app.streamlit_app.summarize_text", return_value="Test summary"):
            desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert desc == "Test summary"
        assert use_case == "Testing/Verification"

    @pytest.mark.happy_path
    def test_small_file_utility_use_case(self, tmp_path):
        """Test: File with 'util' in name triggers 'Utility/Helper Function' use case."""
        file_path = tmp_path / "my_util.py"
        content = "def util_func():\n    pass\n" * 20
        file_path.write_text(content)
        with mock.patch("streamlit_app.streamlit_app.summarize_text", return_value="Util summary"):
            desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert desc == "Util summary"
        assert use_case == "Utility/Helper Function"

    @pytest.mark.happy_path
    def test_small_file_helper_use_case(self, tmp_path):
        """Test: File with 'helper' in name triggers 'Utility/Helper Function' use case."""
        file_path = tmp_path / "string_helper.py"
        content = "def help():\n    pass\n" * 20
        file_path.write_text(content)
        with mock.patch("streamlit_app.streamlit_app.summarize_text", return_value="Helper summary"):
            desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert desc == "Helper summary"
        assert use_case == "Utility/Helper Function"

    # --- EDGE CASES ---

    @pytest.mark.edge_case
    def test_file_too_large(self, tmp_path):
        """Test: File larger than 1MB returns 'File too large for automatic description.'"""
        file_path = tmp_path / "bigfile.py"
        # Write a file just over 1MB
        file_path.write_bytes(b"x" * (1 * 1024 * 1024 + 1))
        desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert desc == "File too large for automatic description."
        assert use_case == "N/A"

    @pytest.mark.edge_case
    def test_empty_file(self, tmp_path):
        """Test: Empty file returns 'File is empty.' and use_case 'N/A'."""
        file_path = tmp_path / "empty.py"
        file_path.write_text("")
        desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert desc == "File is empty."
        assert use_case == "N/A"

    @pytest.mark.edge_case
    def test_file_not_found(self):
        """Test: Non-existent file returns error message."""
        fake_path = "/tmp/this_file_does_not_exist_123456.py"
        desc, use_case = get_llm_response(fake_path, "irrelevant")
        assert desc == "Error: File not found"
        assert use_case == "Error: File not found"

    @pytest.mark.edge_case
    def test_oserror_on_getsize(self, tmp_path):
        """Test: OSError during os.path.getsize returns error message."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")
        with mock.patch("os.path.getsize", side_effect=OSError("Permission denied")):
            desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert "Error accessing file" in desc
        assert "Error:" in use_case

    @pytest.mark.edge_case
    def test_oserror_on_open(self, tmp_path):
        """Test: OSError during open returns error message."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")
        with mock.patch("builtins.open", side_effect=OSError("Disk error")):
            desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert "Error accessing file" in desc or "Error processing file" in desc
        assert "Error:" in use_case

    @pytest.mark.edge_case
    def test_unexpected_exception(self, tmp_path):
        """Test: Unexpected exception during processing returns error message."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")
        # Patch summarize_text to raise an exception
        with mock.patch("streamlit_app.streamlit_app.summarize_text", side_effect=Exception("Unexpected!")):
            desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert "Error processing file" in desc
        assert "Error:" in use_case

    @pytest.mark.edge_case
    def test_file_with_non_utf8_content(self, tmp_path):
        """Test: File with non-UTF8 content is read with errors='ignore' and handled."""
        file_path = tmp_path / "binaryfile.py"
        # Write some bytes that are not valid UTF-8
        file_path.write_bytes(b"\xff\xfe\xfd" + b"def foo():\n    pass\n" * 20)
        # Should not raise, and should call summarize_text with a string
        with mock.patch("streamlit_app.streamlit_app.summarize_text", return_value="Summary for binary"):
            desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert desc == "Summary for binary"
        assert use_case == "Core Logic/Component"

    @pytest.mark.edge_case
    def test_file_with_only_whitespace(self, tmp_path):
        """Test: File with only whitespace is treated as empty."""
        file_path = tmp_path / "whitespace.py"
        file_path.write_text("   \n\t  ")
        desc, use_case = get_llm_response(str(file_path), "irrelevant")
        assert desc == "File is empty."
        assert use_case == "N/A"