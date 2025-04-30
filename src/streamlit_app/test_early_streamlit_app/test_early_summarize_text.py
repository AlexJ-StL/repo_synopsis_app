# test_streamlit_app_summarize_text.py

import pytest
from unittest.mock import patch, MagicMock

# Import the function to test
from streamlit_app.streamlit_app import summarize_text

@pytest.mark.usefixtures("mock_get_summarizer")
class TestSummarizeText:
    """Unit tests for summarize_text function in streamlit_app.streamlit_app."""

    # --- Fixtures ---

    @pytest.fixture(autouse=True)
    def mock_get_summarizer(self):
        """
        Patch get_summarizer before each test.
        """
        with patch("streamlit_app.streamlit_app.get_summarizer") as mock:
            yield mock

    # --- Happy Path Tests ---

    @pytest.mark.happy
    def test_summarize_text_normal(self, mock_get_summarizer):
        """
        Test that summarize_text returns the summary from the summarizer for normal input.
        """
        # Arrange
        input_text = " ".join(["This is a sentence."] * 40)  # 40 sentences, >30 words
        expected_summary = "This is the summary."
        mock_summarizer = MagicMock(return_value=[{"summary_text": expected_summary}])
        mock_get_summarizer.return_value = mock_summarizer

        # Act
        result = summarize_text(input_text, max_length=60)

        # Assert
        assert result == expected_summary
        mock_summarizer.assert_called_once()
        args, kwargs = mock_summarizer.call_args
        assert args[0] == input_text
        assert kwargs["max_length"] <= 60
        assert kwargs["min_length"] >= 10

    @pytest.mark.happy
    def test_summarize_text_dynamic_length(self, mock_get_summarizer):
        """
        Test that summarize_text dynamically adjusts max_length and min_length based on input.
        """
        input_text = " ".join(["word"] * 100)  # 100 words
        expected_summary = "Short summary."
        mock_summarizer = MagicMock(return_value=[{"summary_text": expected_summary}])
        mock_get_summarizer.return_value = mock_summarizer

        result = summarize_text(input_text, max_length=80)
        # max_length should be min(80, max(30, 100//2)) = 50
        # min_length should be max(10, 50//3) = 16

        mock_summarizer.assert_called_once()
        _, kwargs = mock_summarizer.call_args
        assert kwargs["max_length"] == 50
        assert kwargs["min_length"] == 16
        assert result == expected_summary

    # --- Edge Case Tests ---

    @pytest.mark.edge
    def test_summarize_text_empty_string(self, mock_get_summarizer):
        """
        Test that summarize_text returns empty string for empty input.
        """
        result = summarize_text("")
        assert result == ""
        mock_get_summarizer.assert_not_called()

    @pytest.mark.edge
    def test_summarize_text_short_text(self, mock_get_summarizer):
        """
        Test that summarize_text returns empty string for text with less than 30 words.
        """
        input_text = "This is a short text with less than thirty words."
        result = summarize_text(input_text)
        assert result == ""
        mock_get_summarizer.assert_not_called()

    @pytest.mark.edge
    def test_summarize_text_too_long_text(self, mock_get_summarizer):
        """
        Test that summarize_text returns empty string for text longer than 100,000 characters.
        """
        input_text = "word " * 20001  # Each 'word ' is 5 chars, so >100,000 chars
        result = summarize_text(input_text)
        assert result == ""
        mock_get_summarizer.assert_not_called()

    @pytest.mark.edge
    def test_summarize_text_summarizer_returns_none(self, mock_get_summarizer):
        """
        Test that summarize_text returns empty string if get_summarizer returns None.
        """
        input_text = " ".join(["word"] * 40)
        mock_get_summarizer.return_value = None
        result = summarize_text(input_text)
        assert result == ""

    @pytest.mark.edge
    def test_summarize_text_summarizer_returns_unexpected_format(self, mock_get_summarizer, capsys):
        """
        Test that summarize_text returns empty string if summarizer returns unexpected format.
        """
        input_text = " ".join(["word"] * 40)
        mock_get_summarizer.return_value = MagicMock(return_value="not a list")
        result = summarize_text(input_text)
        assert result == ""
        captured = capsys.readouterr()
        assert "Unexpected summarizer output format" in captured.out

    @pytest.mark.edge
    def test_summarize_text_summarizer_raises_exception(self, mock_get_summarizer, capsys):
        """
        Test that summarize_text returns empty string if summarizer raises an exception.
        """
        input_text = " ".join(["word"] * 40)
        def raise_exc(*a, **kw):
            raise RuntimeError("Summarizer failed")
        mock_get_summarizer.return_value = raise_exc
        result = summarize_text(input_text)
        assert result == ""
        captured = capsys.readouterr()
        assert "Summarization error for text starting with" in captured.out

    @pytest.mark.edge
    def test_summarize_text_summarizer_returns_empty_list(self, mock_get_summarizer):
        """
        Test that summarize_text returns empty string if summarizer returns an empty list.
        """
        input_text = " ".join(["word"] * 40)
        mock_get_summarizer.return_value = MagicMock(return_value=[])
        result = summarize_text(input_text)
        assert result == ""

    @pytest.mark.edge
    def test_summarize_text_summarizer_returns_list_without_dict(self, mock_get_summarizer):
        """
        Test that summarize_text returns empty string if summarizer returns a list without a dict.
        """
        input_text = " ".join(["word"] * 40)
        mock_get_summarizer.return_value = MagicMock(return_value=[42])
        result = summarize_text(input_text)
        assert result == ""

    @pytest.mark.edge
    def test_summarize_text_summarizer_returns_dict_without_summary_text(self, mock_get_summarizer):
        """
        Test that summarize_text returns empty string if summarizer returns a dict without 'summary_text'.
        """
        input_text = " ".join(["word"] * 40)
        mock_get_summarizer.return_value = MagicMock(return_value=[{"not_summary": "foo"}])
        result = summarize_text(input_text)
        assert result == ""