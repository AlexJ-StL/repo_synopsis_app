import streamlit as st
import os
import json
import datetime
from typing import Optional, List, Dict, Tuple
from functools import lru_cache
from transformers import pipeline
# You'll need to install transformers: pip install transformers


@lru_cache(maxsize=1)
def get_summarizer():
    """Initialize and cache the summarization pipeline."""
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device="cpu"  # Explicitly set device
    )


def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarizes text using a pre-trained summarization model."""
    if not text:
        return ""  # Return empty string for empty text

    if len(text.split()) < 30:  # Don't summarize very short texts
        return text

    # Calculate dynamic max_length
    input_length = len(text.split())
    max_length = min(max_length, max(30, input_length // 2))
    min_length = max(10, max_length // 3)

    try:
        summarizer = get_summarizer()
        summaries = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        if summaries:
            summary = summaries[0]["summary_text"]
        else:
            summary = text  # Or handle the empty summary case differently
        return summary
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return text  # Return original text if summarization fails


def traverse_directory(directory_path: str) -> List[str]:
    items = []
    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                items.append(os.path.join(root, file))
        return items
    except (PermissionError, OSError) as e:
        st.error(f"Error traversing directory: {e}")
        return []


def generate_directory_tree(directory_path: str) -> str:
    """Generate a tree-like structure of the directory."""
    try:
        tree = []
        for root, dirs, files in os.walk(directory_path):
            level = root.replace(directory_path, '').count(os.sep)
            indent = '  ' * level
            tree.append(f"{indent}{os.path.basename(root)}/")
            for file in files:
                tree.append(f"{indent}  {file}")
        return '\n'.join(tree)
    except OSError:
        return ""  # Return empty string on error


def get_file_language(file_path: str) -> str:
    """Determine programming language based on file extension."""
    extension_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.md': 'Markdown',
        '.json': 'JSON',
        '.xml': 'XML',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.java': 'Java',
        '.h': 'C/C++',
        '.cpp': 'C++',
        '.c': 'C',
        '.cs': 'C#',
        '.go': 'Go',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.rs': 'Rust',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.m': 'Objective-C',
        '.r': 'R'
    }

    ext = os.path.splitext(file_path.lower())[1]
    return extension_map.get(ext, 'Unknown')


def get_llm_response(file_path: str, llm_provider: str) -> Tuple[str, str]:
    """Simplified LLM response (placeholder - replace with actual API call)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            description = summarize_text(content)  # Use summarization
            use_case = "Placeholder use case"  # Replace with better logic
        return description, use_case
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"


def process_repo(
    repo_path: str,
    include_options: Dict[str, bool],
    llm_provider: str
) -> Dict[str, object]:
    if not repo_path:
        return {
                "repo_path": repo_path,
                "error": "Empty repo path provided"
            }
    repo_data = {"repo_path": repo_path, "files": []}
    try:
        items = traverse_directory(repo_path)
        languages = set()

        for item_path in items:
            if os.path.isfile(item_path):
                language = get_file_language(item_path)
                languages.add(language)
                file_data = {"path": item_path, "language": language}

                if include_options["token_count"] and language != "Unknown":
                    try:
                        with open(item_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            file_data["token_count"] = len(content.split())
                    except Exception:
                        file_data["token_count"] = "Unable to read file"

                if include_options[
                    "descriptions"
                ] or include_options[
                    "use_cases"
                ]:
                    description, use_case = get_llm_response(
                        item_path,
                        llm_provider
                    )
                    file_data["description"] = description
                    file_data["use_case"] = use_case
                repo_data["files"].append(file_data)

        repo_data["languages"] = list(languages)
        return repo_data
    except Exception as e:
        st.error(f"Error processing repository {repo_path}: {e}")
        return {"repo_path": repo_path, "error": str(e)}


def handle_directory_error(directory_path: str) -> bool:
    """Validate directory path and handle errors."""
    if not directory_path:
        st.error("Please enter a directory path.")
        return False

    if not os.path.exists(directory_path):
        st.error(f"Directory does not exist: {directory_path}")
        return False

    if not os.path.isdir(directory_path):
        st.error(f"Path is not a directory: {directory_path}")
        return False

    try:
        os.listdir(directory_path)
        return True

    except PermissionError:
        st.error(f"Permission denied accessing directory: {directory_path}")
        return False
    except OSError as e:
        st.error(f"Error accessing directory: {e}")
        return False


def log_event(directory_path: str, message: str) -> None:
    """Log events with timestamp."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file_path = os.path.join(directory_path, "event_log.txt")

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {message}\n")
    except Exception as e:
        st.error(f"Error writing to log file: {e}")


def save_synopsis(directory_path: str, content: str) -> bool:
    """Save the synopsis to a file."""
    if not content:
        st.error("No content to save.")
        return False
    try:
        file_path = os.path.join(directory_path, "synopsis.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        st.success(f"Synopsis saved to {file_path}")
        return True
    except Exception as e:
        st.error(f"Error saving synopsis: {e}")
        return False


def generate_synopsis(
    directory_path: str,
    include_tree: bool = True,
    include_descriptions: bool = True,
    include_token_count: bool = True,
    include_use_cases: bool = True,
    llm_provider: str = "Groq",
) -> Optional[str]:
    """Generate and save synopsis directly."""
    if not handle_directory_error(directory_path):
        return None

    try:
        items = traverse_directory(directory_path)
        if not items:
            st.warning(
                "No items found in directory. Generating empty synopsis."
            )
            return ""  # Return empty string, not None

        synopsis = ""
        languages = set()

        if include_tree:
            synopsis += "## Directory Tree\n"
            tree = generate_directory_tree(directory_path)
            synopsis += tree + os.linesep

        if include_descriptions or include_token_count or include_use_cases:
            synopsis += "## Item Details\n"
            for item_path in items:
                if os.path.isfile(item_path):
                    language = get_file_language(item_path)
                    languages.add(language)
                    synopsis += (
                        f"- **File:** {item_path}, "
                        f"**Language:** {language}\n"
                    )

                    if include_token_count and language != "Unknown":
                        try:
                            with open(item_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                token_count = len(content.split())
                                synopsis += (
                                    f"  - **Token Count:**"
                                    f"{token_count}\n"
                                )
                        except Exception:
                            synopsis += (
                                "  - **Token Count:** "
                                "Unable to read file\n"
                            )

                    if include_descriptions or include_use_cases:
                        description, use_case = get_llm_response(
                            item_path,
                            llm_provider
                        )
                        if include_descriptions:
                            synopsis += f"  - **Description:** {description}\n"
                        if include_use_cases:
                            synopsis += f"  - **Use Case:** {use_case}\n"

        if languages:
            synopsis = f"Languages used: {', '.join(languages)}\n\n" + synopsis

        if save_synopsis(directory_path, synopsis):
            return synopsis
        return None

    except Exception as e:
        st.error(f"Error generating synopsis: {e}")
        log_event(directory_path, f"Error generating synopsis: {e}")
        return None


def main():
    st.title("Repo Synopsis Generator")

    st.subheader("Synopsis Options")
    include_tree = st.checkbox("Directory Tree", value=True)
    include_descriptions = st.checkbox("Descriptions", value=True)
    include_token_count = st.checkbox("Token Count", value=True)
    include_use_cases = st.checkbox("Use Cases", value=True)
    llm_provider = st.selectbox(
        "Select LLM API Provider",
        ("Groq", "Cerberas", "SombaNova", "Gemini"),
    )

    directory_path = st.text_input("Enter directory path:")
    repo_paths = []  # Initialize as an empty list

    if directory_path:  # Only populate if directory_path has a value
        if handle_directory_error(directory_path):
            repo_paths = [
                p for p in os.listdir(directory_path)
                if os.path.isdir(os.path.join(directory_path, p))
            ]
        else:
            repo_paths = []

    selected_repo_paths = st.multiselect(
        "Select repositories",
        repo_paths,
        default=repo_paths
    )

    if st.button("Generate Synopsis"):
        if not directory_path:
            st.warning("Please enter a directory path.")
            return

        if not handle_directory_error(directory_path):
            return

        if not selected_repo_paths:
            st.warning("Please select at least one repository.")
            return

        include_options = {
            "tree": include_tree,
            "descriptions": include_descriptions,
            "token_count": include_token_count,
            "use_cases": include_use_cases,
        }
        repo_data = {}

        for repo_path in [
            os.path.join(directory_path, p) for p in selected_repo_paths
        ]:
            repo_data.update({
                repo_path: process_repo(
                    repo_path, include_options, llm_provider
                )
            })

        # Save the results as a JSON file
        try:
            output_file = os.path.join(directory_path, "repo_synopsis.json")
            with open(output_file, "w") as f:
                json.dump(repo_data, f, indent=4)
            st.success(f"Synopsis saved to {output_file}")
            log_event(directory_path, f"Synopsis saved to {output_file}")

        except Exception as e:
            st.error(f"Error saving synopsis: {e}")
            log_event(directory_path, f"Error saving synopsis: {e}")


if __name__ == "__main__":
    main()
