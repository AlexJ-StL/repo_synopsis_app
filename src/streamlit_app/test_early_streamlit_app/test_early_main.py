import pytest
from unittest.mock import MagicMock, patch, call

# Import the main function
from streamlit_app.streamlit_app import main

@pytest.mark.usefixtures("mocker")
class TestMain:
    @pytest.fixture(autouse=True)
    def setup_streamlit_mocks(self, mocker):
        # Patch all streamlit functions used in main
        self.st_mock = mocker.patch("streamlit_app.streamlit_app.st", autospec=True)
        # Patch st.session_state as a mutable dict
        self.st_mock.session_state = {}

    @pytest.mark.happy_path
    def test_main_happy_path_single_repo_all_options(self, mocker, tmp_path):
        """
        Test main() with a valid base directory containing one repo, all options enabled, and all steps succeed.
        """
        # Setup directory structure
        base_dir = tmp_path / "base"
        repo_dir = base_dir / "repo1"
        repo_dir.mkdir(parents=True)
        (repo_dir / "file1.py").write_text("print('hello world')\n")
        (repo_dir / "file2.txt").write_text("some text file\n")

        # Mock st.sidebar.text_input to return base_dir
        self.st_mock.sidebar.text_input.return_value = str(base_dir)
        # Mock checkboxes and selectbox
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"

        # Mock os.listdir to return the repo
        mocker.patch("os.listdir", return_value=["repo1"])
        # Mock os.path.isdir to return True for repo
        mocker.patch("os.path.isdir", side_effect=lambda p: p == str(repo_dir) or p == str(base_dir))
        # Mock handle_directory_error to always return True
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
        # Mock traverse_directory to return files
        mocker.patch("streamlit_app.streamlit_app.traverse_directory", return_value=[str(repo_dir / "file1.py"), str(repo_dir / "file2.txt")])
        # Mock get_file_language
        mocker.patch("streamlit_app.streamlit_app.get_file_language", side_effect=lambda p: "Python" if p.endswith(".py") else "Text")
        # Mock get_llm_response
        mocker.patch("streamlit_app.streamlit_app.get_llm_response", return_value=("desc", "usecase"))
        # Mock save_synopsis to always succeed
        mocker.patch("streamlit_app.streamlit_app.save_synopsis", return_value=True)
        # Mock generate_directory_tree
        mocker.patch("streamlit_app.streamlit_app.generate_directory_tree", return_value="repo1/\n  file1.py\n  file2.txt")
        # Patch open for writing/reading files
        mocker.patch("builtins.open", mocker.mock_open(read_data="SOME DATA"))
        # Patch json.dump
        mocker.patch("json.dump")
        # Patch os.path.join to behave normally
        mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))
        # Patch os.path.getsize to return small size
        mocker.patch("os.path.getsize", return_value=100)
        # Patch os.path.isfile to True for files
        mocker.patch("os.path.isfile", return_value=True)
        # Patch datetime.datetime.now to a fixed value
        mocker.patch("streamlit_app.streamlit_app.datetime").datetime.now.return_value = mocker.Mock(strftime=lambda fmt: "20240101_120000", total_seconds=lambda: 1.23)

        # Simulate user selection in st.session_state
        self.st_mock.session_state["repo_select"] = ["repo1"]

        # Simulate st.button: first call (not pressed), second call (pressed)
        self.st_mock.button.side_effect = [False, False, True]

        # Run main
        main()

        # Check that st.success was called for both synopsis and JSON
        assert self.st_mock.success.call_count >= 2
        # Check that st.download_button was called for both files
        assert self.st_mock.download_button.call_count >= 2
        # Check that st.markdown was called (for displaying synopsis)
        assert self.st_mock.markdown.called

    @pytest.mark.happy_path
    def test_main_happy_path_no_repos_found(self, mocker, tmp_path):
        """
        Test main() when the base directory is valid but contains no subdirectories (repos).
        """
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        self.st_mock.sidebar.text_input.return_value = str(base_dir)
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        mocker.patch("os.listdir", return_value=[])
        mocker.patch("os.path.isdir", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
        self.st_mock.button.side_effect = [False, False, False]
        main()
        # Should warn about no subdirectories
        assert self.st_mock.warning.call_args
        assert "No subdirectories found" in str(self.st_mock.warning.call_args)

    @pytest.mark.happy_path
    def test_main_happy_path_no_base_directory(self, mocker):
        """
        Test main() when no base directory is entered.
        """
        self.st_mock.sidebar.text_input.return_value = ""
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        self.st_mock.button.side_effect = [False, False, False]
        main()
        # Should show info about entering a base directory
        assert self.st_mock.info.call_args
        assert "base directory path" in str(self.st_mock.info.call_args)

    @pytest.mark.edge_case
    def test_main_edge_case_invalid_base_directory(self, mocker):
        """
        Test main() when an invalid base directory is entered.
        """
        self.st_mock.sidebar.text_input.return_value = "/invalid/path"
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=False)
        self.st_mock.button.side_effect = [False, False, False]
        main()
        # Should show info about entering a valid directory
        assert self.st_mock.info.call_args
        assert "valid and accessible base directory" in str(self.st_mock.info.call_args)

    @pytest.mark.edge_case
    def test_main_edge_case_generate_button_no_selection(self, mocker, tmp_path):
        """
        Test main() when Generate is pressed but no repo is selected.
        """
        base_dir = tmp_path / "base"
        repo_dir = base_dir / "repo1"
        repo_dir.mkdir(parents=True)
        self.st_mock.sidebar.text_input.return_value = str(base_dir)
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        mocker.patch("os.listdir", return_value=["repo1"])
        mocker.patch("os.path.isdir", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
        self.st_mock.session_state["repo_select"] = []
        self.st_mock.button.side_effect = [False, False, True]
        main()
        # Should warn about selecting at least one repo
        assert self.st_mock.warning.call_args
        assert "select at least one repository" in str(self.st_mock.warning.call_args)

    @pytest.mark.edge_case
    def test_main_edge_case_generate_button_invalid_base_directory(self, mocker, tmp_path):
        """
        Test main() when Generate is pressed but base directory is invalid.
        """
        self.st_mock.sidebar.text_input.return_value = "/invalid/path"
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=False)
        self.st_mock.session_state["repo_select"] = ["repo1"]
        self.st_mock.button.side_effect = [False, False, True]
        main()
        # Should error about valid base directory
        assert self.st_mock.error.call_args
        assert "Please enter a valid base directory" in str(self.st_mock.error.call_args)

    @pytest.mark.edge_case
    def test_main_edge_case_process_repo_error(self, mocker, tmp_path):
        """
        Test main() when process_repo returns an error for a repo.
        """
        base_dir = tmp_path / "base"
        repo_dir = base_dir / "repo1"
        repo_dir.mkdir(parents=True)
        self.st_mock.sidebar.text_input.return_value = str(base_dir)
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        mocker.patch("os.listdir", return_value=["repo1"])
        mocker.patch("os.path.isdir", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
        self.st_mock.session_state["repo_select"] = ["repo1"]
        self.st_mock.button.side_effect = [False, False, True]
        # Patch process_repo to return error
        mocker.patch("streamlit_app.streamlit_app.process_repo", return_value={
            "repo_path": str(repo_dir),
            "files": [],
            "languages": None,
            "error": "Some error"
        })
        # Patch log_event to do nothing
        mocker.patch("streamlit_app.streamlit_app.log_event")
        main()
        # Should warn about error in repo
        assert self.st_mock.warning.call_args
        assert "with error" in str(self.st_mock.warning.call_args)
        # Should show st.error for "Some repositories encountered errors"
        assert any("encountered errors" in str(call_arg) for call_arg in self.st_mock.error.call_args_list)

    @pytest.mark.edge_case
    def test_main_edge_case_save_synopsis_fails(self, mocker, tmp_path):
        """
        Test main() when save_synopsis fails (returns False).
        """
        base_dir = tmp_path / "base"
        repo_dir = base_dir / "repo1"
        repo_dir.mkdir(parents=True)
        (repo_dir / "file1.py").write_text("print('hello world')\n")
        self.st_mock.sidebar.text_input.return_value = str(base_dir)
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        mocker.patch("os.listdir", return_value=["repo1"])
        mocker.patch("os.path.isdir", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.traverse_directory", return_value=[str(repo_dir / "file1.py")])
        mocker.patch("streamlit_app.streamlit_app.get_file_language", return_value="Python")
        mocker.patch("streamlit_app.streamlit_app.get_llm_response", return_value=("desc", "usecase"))
        mocker.patch("streamlit_app.streamlit_app.save_synopsis", return_value=False)
        mocker.patch("streamlit_app.streamlit_app.generate_directory_tree", return_value="repo1/\n  file1.py")
        mocker.patch("builtins.open", mocker.mock_open(read_data="SOME DATA"))
        mocker.patch("json.dump")
        mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))
        mocker.patch("os.path.getsize", return_value=100)
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.datetime").datetime.now.return_value = mocker.Mock(strftime=lambda fmt: "20240101_120000", total_seconds=lambda: 1.23)
        self.st_mock.session_state["repo_select"] = ["repo1"]
        self.st_mock.button.side_effect = [False, False, True]
        main()
        # Should not call st.success for synopsis file
        assert not any("Combined synopsis saved" in str(call_arg) for call_arg in self.st_mock.success.call_args_list)

    @pytest.mark.edge_case
    def test_main_edge_case_json_save_fails(self, mocker, tmp_path):
        """
        Test main() when saving the JSON file fails.
        """
        base_dir = tmp_path / "base"
        repo_dir = base_dir / "repo1"
        repo_dir.mkdir(parents=True)
        (repo_dir / "file1.py").write_text("print('hello world')\n")
        self.st_mock.sidebar.text_input.return_value = str(base_dir)
        self.st_mock.sidebar.checkbox.side_effect = [True, True, True, True]
        self.st_mock.sidebar.selectbox.return_value = "Groq"
        mocker.patch("os.listdir", return_value=["repo1"])
        mocker.patch("os.path.isdir", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.handle_directory_error", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.traverse_directory", return_value=[str(repo_dir / "file1.py")])
        mocker.patch("streamlit_app.streamlit_app.get_file_language", return_value="Python")
        mocker.patch("streamlit_app.streamlit_app.get_llm_response", return_value=("desc", "usecase"))
        mocker.patch("streamlit_app.streamlit_app.save_synopsis", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.generate_directory_tree", return_value="repo1/\n  file1.py")
        # Patch open to raise exception on JSON save
        def open_side_effect(path, mode, encoding=None):
            if path.endswith(".json"):
                raise IOError("Cannot write JSON")
            return MagicMock()
        mocker.patch("builtins.open", side_effect=open_side_effect)
        mocker.patch("json.dump")
        mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))
        mocker.patch("os.path.getsize", return_value=100)
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("streamlit_app.streamlit_app.datetime").datetime.now.return_value = mocker.Mock(strftime=lambda fmt: "20240101_120000", total_seconds=lambda: 1.23)
        self.st_mock.session_state["repo_select"] = ["repo1"]
        self.st_mock.button.side_effect = [False, False, True]
        # Patch log_event to do nothing
        mocker.patch("streamlit_app.streamlit_app.log_event")
        main()
        # Should call st.error for JSON save
        assert any("Error saving detailed JSON data" in str(call_arg) for call_arg in self.st_mock.error.call_args_list)