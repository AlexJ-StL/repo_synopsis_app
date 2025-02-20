import streamlit as st
import os
import datetime
from typing import Optional

def handle_directory_error(directory_path: str) -> Optional[bool]:
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

def save_synopsis(directory_path: str, synopsis: str) -> bool:
    """Save synopsis directly to source directory."""
    try:
        file_path = os.path.join(directory_path, "repo_synopsis.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(synopsis)
        st.success(f"Synopsis saved to {file_path}")
        log_event(directory_path, f"Synopsis saved to {file_path}")
        return True
    except Exception as e:
        st.error(f"Error saving synopsis: {e}")
        log_event(directory_path, f"Error saving synopsis: {e}")
        return False

def log_event(directory_path: str, message: str) -> None:
    """Log events with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_path = os.path.join(directory_path, "event_log.txt")
    try:
        with open(log_file_path, "a") as f:
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
        if not items:
            st.error("No items found in directory")
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
                            synopsis += f"  - **Token Count:** Unable to read file\n"

                    if include_descriptions or include_use_cases:
                        description, use_case = get_llm_response(item_path, llm_provider)
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

if __name__ == "__main__":
    main()
