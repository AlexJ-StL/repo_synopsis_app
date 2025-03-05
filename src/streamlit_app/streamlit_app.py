import streamlit as st
import os
import datetime
from typing import Optional, List, Tuple


def traverse_directory(directory_path: str) -> List[str]:
    """Recursively traverse directory and return list of file paths."""
    items = []
    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                items.append(os.path.join(root, file))
        return items
    except (PermissionError, OSError):
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

def get_llm_response(file_path: str, llm_provider: str) -> Tuple[str, str]:
    """Get description and use case from LLM API."""
    try:
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path")

        if not llm_provider or llm_provider not in ("Groq", "Cerberas", "SombaNova", "Gemini"):
            return ("Error: Invalid LLM provider", "Error: Invalid LLM provider")

        if llm_provider == "Groq":
            description = f"Sample description for {os.path.basename(file_path)}"
            use_case = f"Sample use case for {os.path.basename(file_path)}"
        else:
            description = (f"Alternative description for "
                f"{os.path.basename(file_path)}")
            use_case = (f"Alternative use case for "
                f"{os.path.basename(file_path)}")
        return description, use_case
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"
        for root, _, files in os.walk(directory_path):
            for file in files:
                items.append(os.path.join(root, file))
        return items
    except (PermissionError, OSError):
        return []
import streamlit as st
def generate_directory_tree(directory_path: str) -> str:
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path")

        if not llm_provider or llm_provider not in ("Groq", "Cerberas", "SombaNova", "Gemini"):
            return ("Error: Invalid LLM provider", "Error: Invalid LLM provider")

            use_case = f"Sample use case for {os.path.basename(file_path)}"
        else:
            description = (f"Alternative description for "
                f"{os.path.basename(file_path)}")
            use_case = (f"Alternative use case for "
                f"{os.path.basename(file_path)}")
        return description, use_case
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"
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


def get_file_language(file_path: str) -> str:
    """Determine programming language based on file extension."""
    extension_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        return items
        '.md': 'Markdown',
        return ""  # Return empty string on error
        if synopsis:
            st.success("Synopsis generated and saved successfully!")


if __name__ == "__main__":
    main(
    directory_path = st.text_input("Enter directory path:")

    if st.button("Generate Synopsis"):
        synopsis = generate_synopsis(
            directory_path,
            include_tree,
            include_descriptions,
            include_token_count,
            include_use_cases,
            llm_provider
        )
        if synopsis:
            st.success("Synopsis generated and saved successfully!")


if __nam'.json': 'JSON',

        '.xml': 'XML',
        '.yaml': 'YAML',
        '.yml': 'YAML'
    }

    ext = os.path.splitext(file_path.lower())[1]
    return extension_map.get(ext, 'Unknown')


def get_llm_response(file_path: str, llm_provider: str) -> Tuple[str, str]:
    """Get description and use case from LLM API."""
    try:
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path")
            use_case = f"Alternative use case for {os.path.basename(file_path)}"
        return description, use_case

        if not llm_provider or llm_provider not in ("Groq", "Cerberas", "SombaNova", "Gemini"):
            return "Error: Invalid LLM provider", "Error: Invalid LLM provider"

        if llm_provider == "Groq":
            description = f"Sample description for {os.path.basename(file_path)}"
            use_case = f"Sample use case for {os.path.basename(file_path)}"
        else:
            description = f"Alternative description for {os.path.basename(file_path)}"
            use_case = f"Alternative use case for {os.path.basename(file_path)}"
        return description, use_case
        '.java': 'Java',
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"
        '.h': 'C/C++',


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
        st.error("Please enter a directory path.")
        st.error(f"Error accessing directory: {e}")
        return False


def save_synopsis(directory_path: str, synopsis: str) -> bool:
    """Save synopsis directly to source directory."""
    try:
        if not synopsis or synopsis.strip() == "":
        st.error(f"Path is not a directory: {directory_path}")
            st.error("Cannot save empty synopsis")
            log_event(
    try:
                directory_path,
                "Error: Attempted to save empty synopsis"
    except PermissionError:
            )
            return False

        file_path = os.path.join(directory_path, "repo_synopsis.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(synopsis)
        st.success(f"Synopsis saved to {file_path}")
    """Save synopsis directly to source directory."""
        log_event(directory_path, f"Synopsis saved to {file_path}")
        return True
            log_event(
    except Exception as e:
        st.error(f"Error saving synopsis: {e}")
        log_event(directory_path, f"Error saving synopsis: {e}")
        return False


def log_event(directory_path: str, message: str) -> None:
    """Log events with timestamp."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

    except Exception as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file_path = os.path.join(directory_path, "event_log.txt")

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {message}\n")
    except Exception as e:
        st.error(f"Error writing to log file: {e}")



def generate_synopsis(
    directory_path: str,
    include_tree: bool,
    include_descriptions: bool,
    include_token_count: bool,
    include_use_cases: bool,
    llm_provider: str
) -> Optional[str]:
    """Generate and save synopsis directly."""
    if not handle_directory_error(directory_path):
        return None

    try:
        items = traverse_directory(directory_path)
    llm_provider: str
        if not items:
            st.error("No items found in directory")
            return None
        return None

        synopsis = ""
        languages = set()

        if include_tree:
            synopsis += "## Directory Tree\n"

            tree = generate_directory_tree(directory_path)
            synopsis += tree.replace("\\n", "\n") + "\n"


        if include_descriptions or include_token_count or include_use_cases:
            synopsis += "## Item Details\n"
            for item_path in items:
                if os.path.isfile(item_path):
                    language = get_file_language(item_path)
                    languages.add(language)
                    synopsis += f"- **File:** {item_path}, **Language:** {language}\n"

                    if include_token_count and language != "Unknown":
                        try:
                            with open(item_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                token_count = len(content.split())
                                synopsis += f"  - **Token Count:** {token_count}\n"
                        except Exception:
                                content = f.read()
                            synopsis += "  - **Token Count:** Unable to read file\n"  # Fixed f-string issue

                    if include_descriptions or include_use_cases:
                        description, use_case = get_llm_response(
                            item_path,
                            llm_provider
                        )
                        if include_descriptions:
                            synopsis += f"  - **Description:** {description}\n"
                        if include_use_cases:
                            synopsis += f"  - **Use Case:** {use_case}\n"

                        if include_use_cases:
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

    llm_provider = st.selectbox(
    )
        "Select LLM API Provider",
        ("Groq", "Cerberas", "SombaNova", "Gemini"),
    )ini"),
    )

    directory_path = st.text_input("Enter directory path:")

    f st.button("Generate Syopss:
        synopsis = generate_synopsis(
            directory_path
         include_tree,
        )
           include_descriptions,
            include_token_count,

            include_use_cases,
            llm_provider
        e__ == "__main__":
    main()
if __name__ == "__main__":
    main()
