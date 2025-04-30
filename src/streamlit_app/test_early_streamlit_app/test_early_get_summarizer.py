import pytest
from unittest.mock import patch, MagicMock
from transformers import Pipeline
from src.streamlit_app.streamlit_app import get_summarizer

@pytest.mark.usefixtures("mock_pipeline")
class TestGetSummarizer:
    @pytest.fixture
    def mock_pipeline(self):
        """Fixture to mock the transformers pipeline."""
        with patch('src.streamlit_app.streamlit_app.pipeline') as mock:
            yield mock

    @pytest.mark.happy_path
    def test_get_summarizer_success(self, mock_pipeline):
        """Test that get_summarizer returns a Pipeline object when successful."""
        mock_pipeline.return_value = MagicMock(spec=Pipeline)

        summarizer = get_summarizer()

        assert isinstance(summarizer, Pipeline)
        mock_pipeline.assert_called_once_with(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cpu"
        )

    @pytest.mark.edge_case
    def test_get_summarizer_failure(self, mock_pipeline):
        """Test that get_summarizer raises RuntimeError on pipeline initialization failure."""
        mock_pipeline.side_effect = Exception("Initialization error")

        with pytest.raises(RuntimeError, match="Failed to load summarization model: Initialization error"):
            get_summarizer()

    @pytest.mark.happy_path
    def test_get_summarizer_cache(self, mock_pipeline):
        """Test that get_summarizer uses caching to avoid re-initialization."""
        mock_pipeline.return_value = MagicMock(spec=Pipeline)

        summarizer1 = get_summarizer()
        summarizer2 = get_summarizer()

        assert summarizer1 is summarizer2
        mock_pipeline.assert_called_once()

# To run these tests, you would typically use the command:
# pytest -v --tb=short