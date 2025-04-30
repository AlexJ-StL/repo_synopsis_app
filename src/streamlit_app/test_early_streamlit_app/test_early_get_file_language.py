# test_streamlit_app_get_file_language.py

import pytest

from streamlit_app.streamlit_app import get_file_language

@pytest.mark.usefixtures()
class TestGetFileLanguage:
    # ------------------- Happy Path Tests -------------------

    @pytest.mark.happy_path
    def test_python_file(self):
        """Test detection of Python file by .py extension."""
        assert get_file_language("main.py") == "Python"

    @pytest.mark.happy_path
    def test_python_notebook_file(self):
        """Test detection of Python Notebook file by .ipynb extension."""
        assert get_file_language("analysis.ipynb") == "Python Notebook"

    @pytest.mark.happy_path
    def test_javascript_file(self):
        """Test detection of JavaScript file by .js extension."""
        assert get_file_language("app.js") == "JavaScript"

    @pytest.mark.happy_path
    def test_typescript_file(self):
        """Test detection of TypeScript file by .ts extension."""
        assert get_file_language("index.ts") == "TypeScript"

    @pytest.mark.happy_path
    def test_html_file(self):
        """Test detection of HTML file by .html extension."""
        assert get_file_language("index.html") == "HTML"

    @pytest.mark.happy_path
    def test_css_file(self):
        """Test detection of CSS file by .css extension."""
        assert get_file_language("style.css") == "CSS"

    @pytest.mark.happy_path
    def test_markdown_file(self):
        """Test detection of Markdown file by .md extension."""
        assert get_file_language("README.md") == "Markdown"

    @pytest.mark.happy_path
    def test_json_file(self):
        """Test detection of JSON file by .json extension."""
        assert get_file_language("data.json") == "JSON"

    @pytest.mark.happy_path
    def test_yaml_file(self):
        """Test detection of YAML file by .yaml extension."""
        assert get_file_language("config.yaml") == "YAML"

    @pytest.mark.happy_path
    def test_yml_file(self):
        """Test detection of YAML file by .yml extension."""
        assert get_file_language("config.yml") == "YAML"

    @pytest.mark.happy_path
    def test_java_file(self):
        """Test detection of Java file by .java extension."""
        assert get_file_language("Main.java") == "Java"

    @pytest.mark.happy_path
    def test_cpp_file(self):
        """Test detection of C++ file by .cpp extension."""
        assert get_file_language("main.cpp") == "C++"

    @pytest.mark.happy_path
    def test_c_file(self):
        """Test detection of C file by .c extension."""
        assert get_file_language("main.c") == "C"

    @pytest.mark.happy_path
    def test_csharp_file(self):
        """Test detection of C# file by .cs extension."""
        assert get_file_language("Program.cs") == "C#"

    @pytest.mark.happy_path
    def test_go_file(self):
        """Test detection of Go file by .go extension."""
        assert get_file_language("main.go") == "Go"

    @pytest.mark.happy_path
    def test_php_file(self):
        """Test detection of PHP file by .php extension."""
        assert get_file_language("index.php") == "PHP"

    @pytest.mark.happy_path
    def test_ruby_file(self):
        """Test detection of Ruby file by .rb extension."""
        assert get_file_language("script.rb") == "Ruby"

    @pytest.mark.happy_path
    def test_rust_file(self):
        """Test detection of Rust file by .rs extension."""
        assert get_file_language("main.rs") == "Rust"

    @pytest.mark.happy_path
    def test_swift_file(self):
        """Test detection of Swift file by .swift extension."""
        assert get_file_language("app.swift") == "Swift"

    @pytest.mark.happy_path
    def test_objective_c_file(self):
        """Test detection of Objective-C file by .m extension."""
        assert get_file_language("main.m") == "Objective-C"

    @pytest.mark.happy_path
    def test_shell_file(self):
        """Test detection of Shell file by .sh extension."""
        assert get_file_language("deploy.sh") == "Shell"

    @pytest.mark.happy_path
    def test_batch_file(self):
        """Test detection of Batch file by .bat extension."""
        assert get_file_language("run.bat") == "Batch"

    @pytest.mark.happy_path
    def test_powershell_file(self):
        """Test detection of PowerShell file by .ps1 extension."""
        assert get_file_language("script.ps1") == "PowerShell"

    @pytest.mark.happy_path
    def test_sql_file(self):
        """Test detection of SQL file by .sql extension."""
        assert get_file_language("query.sql") == "SQL"

    @pytest.mark.happy_path
    def test_dockerfile_with_extension(self):
        """Test detection of Dockerfile by .dockerfile extension."""
        assert get_file_language("myimage.dockerfile") == "Dockerfile"

    @pytest.mark.happy_path
    def test_dockerfile_without_extension(self):
        """Test detection of Dockerfile by filename 'Dockerfile' (no extension)."""
        assert get_file_language("Dockerfile") == "Dockerfile"

    @pytest.mark.happy_path
    def test_makefile_without_extension(self):
        """Test detection of Makefile by filename 'Makefile' (no extension)."""
        assert get_file_language("Makefile") == "Makefile"

    @pytest.mark.happy_path
    def test_terraform_file(self):
        """Test detection of Terraform file by .tf extension."""
        assert get_file_language("main.tf") == "Terraform"

    @pytest.mark.happy_path
    def test_hcl_file(self):
        """Test detection of HCL file by .hcl extension."""
        assert get_file_language("config.hcl") == "HCL"

    @pytest.mark.happy_path
    def test_text_file(self):
        """Test detection of Text file by .txt extension."""
        assert get_file_language("notes.txt") == "Text"

    @pytest.mark.happy_path
    def test_file_with_uppercase_extension(self):
        """Test detection is case-insensitive for extension."""
        assert get_file_language("README.MD") == "Markdown"

    @pytest.mark.happy_path
    def test_file_with_path(self):
        """Test detection works with full file paths."""
        assert get_file_language("/home/user/project/main.py") == "Python"

    @pytest.mark.happy_path
    def test_file_with_multiple_dots(self):
        """Test detection works with files with multiple dots in name."""
        assert get_file_language("archive.tar.gz") == "Other"

    # ------------------- Edge Case Tests -------------------

    @pytest.mark.edge_case
    def test_file_with_no_extension(self):
        """Test detection for file with no extension and unrecognized basename."""
        assert get_file_language("LICENSE") == "Unknown"

    @pytest.mark.edge_case
    def test_file_with_empty_string(self):
        """Test detection for empty string as file path."""
        assert get_file_language("") == "Unknown"

    @pytest.mark.edge_case
    def test_file_with_dotfile(self):
        """Test detection for dotfile (e.g., .env) with no extension."""
        assert get_file_language(".env") == "Other"

    @pytest.mark.edge_case
    def test_file_with_hidden_dockerfile(self):
        """Test detection for hidden Dockerfile (e.g., .dockerfile)."""
        assert get_file_language(".dockerfile") == "Dockerfile"

    @pytest.mark.edge_case
    def test_file_with_hidden_makefile(self):
        """Test detection for hidden Makefile (e.g., .makefile)."""
        assert get_file_language(".makefile") == "Other"

    @pytest.mark.edge_case
    def test_file_with_uncommon_extension(self):
        """Test detection for file with uncommon/unknown extension."""
        assert get_file_language("file.unknownext") == "Other"

    @pytest.mark.edge_case
    def test_file_with_uppercase_basename(self):
        """Test detection for uppercase Dockerfile and Makefile (no extension)."""
        assert get_file_language("DOCKERFILE") == "Dockerfile"
        assert get_file_language("MAKEFILE") == "Makefile"

    @pytest.mark.edge_case
    def test_file_with_tricky_name(self):
        """Test detection for file named 'dockerfile.txt' (should be Text)."""
        assert get_file_language("dockerfile.txt") == "Text"

    @pytest.mark.edge_case
    def test_file_with_no_name_only_extension(self):
        """Test detection for file named only as extension (e.g., '.py')."""
        assert get_file_language(".py") == "Python"

    @pytest.mark.edge_case
    def test_file_with_double_extension(self):
        """Test detection for file with double extension (e.g., 'file.tar.gz')."""
        assert get_file_language("file.tar.gz") == "Other"

    @pytest.mark.edge_case
    def test_file_with_directory_in_path(self):
        """Test detection for file in nested directory."""
        assert get_file_language("src/app/main.rs") == "Rust"

    @pytest.mark.edge_case
    def test_file_with_spaces_in_name(self):
        """Test detection for file with spaces in name."""
        assert get_file_language("my script.py") == "Python"

    @pytest.mark.edge_case
    def test_file_with_leading_trailing_spaces(self):
        """Test detection for file with leading/trailing spaces in name."""
        assert get_file_language("  main.py  ") == "Python"

    @pytest.mark.edge_case
    def test_file_with_dot_at_end(self):
        """Test detection for file with dot at the end (e.g., 'file.')."""
        assert get_file_language("file.") == "Other"