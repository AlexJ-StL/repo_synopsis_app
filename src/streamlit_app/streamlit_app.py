import streamlit as st
import os
import datetime
from typing import Optional, Tuple

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
        # Test if directory is readable
        os.listdir(directory_path)
        return True
    except PermissionError:
        st.error(f"Permission denied accessing directory: {directory_path}")
        return False
    except OSError as e:
        st.error(f"Error accessing directory: {e}")
        return False

def safe_file_read(file_path: str) -> Optional[str]:
    """Safely read file contents with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        st.warning(f"Unable to read {file_path} - file may be binary")
        return None
    except PermissionError:
        st.error(f"Permission denied reading file: {file_path}")
        return None
    except OSError as e:
        st.error(f"Error reading file {file_path}: {e}")
        return None

def safe_llm_call(prompt: str, provider: str) -> Tuple[str, str]:
    """Safely make LLM API calls with error handling."""
    try:
        description, use_case = get_llm_response(prompt, provider)
        if not description or not use_case:
            return ("Failed to generate description", "Failed to generate use case")
        return (description, use_case)
    except TimeoutError:
        return ("Error: API request timed out", "Error: API request timed out")
    except ConnectionError:
        return ("Error: Network connection failed", "Error: Network connection failed")
    except Exception as e:
        return (f"Error: {str(e)}", f"Error: {str(e)}")

def safe_synopsis_save(directory_path: str, synopsis: str, custom_save_directory: Optional[str] = None) -> bool:
    """Safely save synopsis with error handling."""
    save_path = custom_save_directory or directory_path

    try:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, "repo_synopsis.md")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(synopsis)

        st.success(f"Synopsis saved to {file_path}")
        log_event(save_path, f"Synopsis saved to {file_path}")
        return True

    except PermissionError:
        st.error(f"Permission denied writing to {save_path}")
        log_event(directory_path, f"Permission denied writing to {save_path}")
        return False
    except OSError as e:
        st.error(f"Error saving synopsis: {e}")
        log_event(directory_path, f"Error saving synopsis: {e}")
        return False

def log_event(directory_path, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_path = (
        os.path.join(directory_path, "event_log.txt")
        if directory_path
        else "event_log.txt"
    )
    try:
        with open(log_file_path, "a") as f:
            f.write(f"{timestamp} - {message}\\n")
    except Exception as e:
        st.write(f"Error writing to log file: {e}")


def generate_directory_tree(directory_path, indent=""):
    tree = ""
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        tree += indent + "- " + item + "\\n"
        if os.path.isdir(item_path):
            tree += generate_directory_tree(item_path, indent + "  ")
    return tree


def get_file_language(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension in [".py"]:
        return "Python"
    elif extension in [".js", ".jsx"]:
        return "JavaScript"
    elif extension in [".ts", ".tsx"]:
        return "TypeScript"
    elif extension in [".java"]:
        return "Java"
    elif extension in [".c", ".cpp"]:
        return "C/C++"
    elif extension in [".go"]:
        return "Go"
    elif extension in [".rs"]:
        return "Rust"
    elif extension in [".php"]:
        return "PHP"
    elif extension in [".html", ".htm"]:
        return "HTML"
    elif extension in [".css"]:
        return "CSS"
    elif extension in [".sql"]:
        return "SQL"
    elif extension in [".sh", ".bash"]:
        return "Shell"
    elif extension in [".md"]:
        return "Markdown"
    else:
        return "Unknown"


def get_llm_response(item_path, llm_provider):
    # Placeholder for LLM integration
    if llm_provider == "Groq":
        description = "Groq description placeholder"
        use_case = "Groq use case placeholder"
    elif llm_provider == "Cerberas":
        description = "Cerberas description placeholder"
        use_case = "Cerberas use case placeholder"
    elif llm_provider == "SombaNova":
        description = "SombaNova description placeholder"
        use_case = "SombaNova use case placeholder"
    elif llm_provider == "Gemini":
        description = "Gemini description placeholder"
        use_case = "Gemini use case placeholder"
    else:
        description = "Default description placeholder"
        use_case = "Default use case placeholder"
    return description, use_case


def traverse_directory(directory_path):
    items = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        items.append(item_path)
        if os.path.isdir(item_path):
            items.extend(traverse_directory(item_path))
    return items


def generate_synopsis(
    directory_path,
    include_tree,
    include_descriptions,
    include_token_count,
    include_use_cases,
    llm_provider,
):
    print(f"DEBUG: Starting generate_synopsis with path: {directory_path}")  # Debug log
    st.write(f"Generating synopsis for: {directory_path}")

    if not directory_path:
        print("DEBUG: No directory path provided")  # Debug log
        st.write("Please enter a directory path.")
        return

    if not os.path.isdir(directory_path):
        print(f"DEBUG: Invalid directory path: {directory_path}")  # Debug log
        st.error(f"Error: The specified directory does not exist: {directory_path}")
        return

    try:
        items = traverse_directory(directory_path)
        print(f"DEBUG: Found {len(items)} items")  # Debug log
    except Exception as e:
        print(f"DEBUG: Error in traverse_directory: {e}")  # Debug log
        st.error(f"Error accessing directory: {e}")
        return

    if not items:
        print("DEBUG: No items found in directory")  # Debug log
        return

    st.write(f"Found {len(items)} items in the directory.")

    synopsis = ""
    languages = set()

    try:
        if include_tree:
            print("DEBUG: Generating directory tree")  # Debug log
            synopsis += "## Directory Tree\n"
            tree = generate_directory_tree(directory_path)
            synopsis += tree.replace("\\\\n", "\\n") + "\n"

        if include_descriptions or include_token_count or include_use_cases:
            print("DEBUG: Processing items for details")  # Debug log
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
                        except Exception as e:
                            print(f"DEBUG: Error reading file {item_path}: {e}")  # Debug log
                            synopsis += f"  - **Error reading file {item_path}:** {e}\n"

                    if include_descriptions or include_use_cases:
                        description, use_case = get_llm_response(item_path, llm_provider)
                        if include_descriptions:
                            synopsis += f"  - **Description:** {description}\n"
                        if include_use_cases:
                            synopsis += f"  - **Use Case:** {use_case}\n"

                elif include_descriptions:
                    synopsis += f"- **Directory:** {item_path}\n"
                    if include_descriptions or include_use_cases:
                        description, use_case = get_llm_response(item_path, llm_provider)
                        if include_descriptions:
                            synopsis += f"  - **Description:** {description}\n"
                        if include_use_cases:
                            synopsis += f"  - **Use Case:** {use_case}\n"

        if languages:
            synopsis = f"Languages used: {', '.join(languages)}\n\n" + synopsis

    except Exception as e:
        print(f"DEBUG: Error generating synopsis content: {e}")  # Debug log
        st.error(f"Error generating synopsis: {e}")
        return

    print("DEBUG: Creating UI elements")  # Debug log

    # Create a container for the synopsis preview and save options
    with st.container():
        st.subheader("Synopsis Preview")
        synopsis_review = st.text_area("Synopsis", value=synopsis, height=300)

        col1, col2 = st.columns(2)
        with col1:
            save_in_source_directory = st.checkbox("Save in source directory", value=True)

        custom_save_directory = None
        if not save_in_source_directory:
            with col2:
                custom_save_directory = st.text_input("Enter custom save directory:")

        if st.button("Save Synopsis", key="save_button"):
            print("DEBUG: Save button clicked")  # Debug log
            try:
                save_path = directory_path if save_in_source_directory else (custom_save_directory or directory_path)
                print(f"DEBUG: Save path determined: {save_path}")  # Debug log

                # Create the full file path
                file_path = os.path.join(save_path, "repo_synopsis.md")
                print(f"DEBUG: Full file path: {file_path}")  # Debug log

                # Ensure the directory exists
                os.makedirs(save_path, exist_ok=True)
                print("DEBUG: Directory created/verified")  # Debug log

                # Save the file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(synopsis_review)
                print("DEBUG: File written successfully")  # Debug log

                st.success(f"Synopsis saved successfully to: {file_path}")
                log_event(save_path, f"Synopsis saved to {file_path}")

            except Exception as e:
                print(f"DEBUG: Error saving file: {e}")  # Debug log
                st.error(f"Failed to save synopsis: {e}")
                log_event(directory_path, f"Error saving synopsis: {e}")

    return synopsis


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

    if st.button("Proceed"):
        if not handle_directory_error(directory_path):
            return

        try:
            synopsis = generate_synopsis(
                directory_path,
                include_tree,
                include_descriptions,
                include_token_count,
                include_use_cases,
                llm_provider
            )

            if synopsis:
                with st.container():
                    st.subheader("Synopsis Preview")
                    synopsis_review = st.text_area("Synopsis", value=synopsis, height=300)

                    col1, col2 = st.columns(2)
                    with col1:
                        save_in_source = st.checkbox("Save in source directory", value=True)

                    custom_save_dir = None
                    if not save_in_source:
                        with col2:
                            custom_save_dir = st.text_input("Enter custom save directory:")

                    if st.button("Save Synopsis"):
                        if not save_in_source and not custom_save_dir:
                            st.error("Please enter a custom save directory")
                            return

                        safe_synopsis_save(directory_path, synopsis_review, custom_save_dir)

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            log_event(directory_path, f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
