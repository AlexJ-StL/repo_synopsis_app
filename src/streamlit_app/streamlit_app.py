import streamlit as st
import os
import json
import datetime
from typing import Optional, List, Dict, Tuple, TypedDict, Any, Union # Added Union
from functools import lru_cache
from transformers import pipeline, Pipeline # Import Pipeline type hint
# You'll need to install transformers: pip install transformers

# --- Type Definitions ---

class FileData(TypedDict, total=False):
    path: str
    language: str
    token_count: Optional[Union[int, str]] # Allow str for error messages
    description: Optional[str]
    use_case: Optional[str]

class RepoData(TypedDict):
    repo_path: str
    files: List[FileData]
    languages: Optional[List[str]]
    error: Optional[str]

# --- Helper Functions (Define before use) ---

@lru_cache(maxsize=1)
def get_summarizer() -> Pipeline:
    """
    Initialize and cache the summarization pipeline using the BART model.

    Returns:
    Pipeline: A summarization pipeline object.

    Raises:
    RuntimeError: If the pipeline initialization fails.
    """
    """Initialize and cache the summarization pipeline."""
    try:
        # Consider adding device selection based on availability (e.g., cuda if available)
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cpu"
        )
    except Exception as e:
        # Handle potential errors during pipeline initialization
        st.error(f"Failed to load summarization model: {e}")
        # You might want to return a dummy function or raise an error
        # depending on how critical summarization is.
        # For now, let it raise, or return None and check later.
        raise RuntimeError(f"Failed to load summarization model: {e}") from e


def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarizes text using a pre-trained summarization model."""
    if not text:
        return ""

    word_count = len(text.split())
    # Avoid summarizing very short text or text likely to cause issues
    if word_count < 30 or len(text) > 100000: # Add a char limit safeguard
        return "" # Return empty string instead of original for consistency

    # Calculate dynamic max_length
    max_length = min(max_length, max(30, word_count // 2))
    min_length = max(10, max_length // 3)

    try:
        summarizer = get_summarizer()
        if summarizer is None: # Check if pipeline loaded
            return ""

        # The pipeline call result type can be complex, check defensively
        summaries_result: Any = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )

        # --- Type Check ---
        # Ensure it's a list and the first element is a dictionary
        if (isinstance(summaries_result, list) and
                summaries_result and # Check if list is not empty
                isinstance(summaries_result[0], dict)):
            """Summarizes the input text using a pre-trained summarization model.
            Parameters:
            text (str): The text to be summarized.
            max_length (int): The maximum length of the summary (default is 150).
            Returns:
            str: The summarized text or an empty string if input is invalid or an error occurs.
            Exceptions:
            May raise exceptions related to model loading or processing, which are caught and handled internally."""
            summary = summaries_result[0].get("summary_text", "") # Use .get for safety
        else:
            # Log unexpected format if needed
            print(f"Unexpected summarizer output format: {type(summaries_result)}")
            summary = "" # Return empty if format is not as expected

        return summary
    except Exception as e:
        print(f"Summarization error for text starting with '{text[:50]}...': {e}")
        return "" # Return empty string on any summarization error


def traverse_directory(directory_path: str) -> List[str]:
    """Walks a directory and returns a list of full file paths."""
    items = []
    try:
        for root, _, files in os.walk(directory_path):
            """Generate a string representation of the directory tree.

            Parameters:
            directory_path (str): The path of the directory to traverse.

            Returns:
            str: A string representation of the directory tree.

            Raises:
            PermissionError: If the program lacks permissions to access a directory.
            OSError: For other OS-related errors encountered during traversal.
            """
            for file in files:
                # Construct full path and normalize (optional but good practice)
                full_path = os.path.normpath(os.path.join(root, file))
                items.append(full_path)
        return items
    except (PermissionError, OSError) as e:
        # Log the error for debugging purposes
        print(f"Error traversing directory {directory_path}: {e}")
        st.warning(f"Could not fully traverse directory due to error: {e}")
        return items # Return whatever was collected so far


def generate_directory_tree(directory_path: str) -> str:
    """Generate a string representation of the directory tree."""
    try:
        tree_lines = []
        # Normalize base path for comparison
        norm_base_path = os.path.normpath(directory_path)
        len_base_path = len(norm_base_path.split(os.sep))

        for root, dirs, files in os.walk(directory_path, topdown=True):
            # Sort directories and files for consistent output
            dirs.sort()
            files.sort()

            norm_root = os.path.normpath(root)
            level = len(norm_root.split(os.sep)) - len_base_path
            indent = "  " * level

            # Add directory entry (if not the root itself)
            if norm_root != norm_base_path:
                tree_lines.append(f"{indent}{os.path.basename(norm_root)}/")

            # Indentation for contents within the current root
            if level == 0:
                content_indent = "  " # Two spaces for items directly in the base directory
            else:
                content_indent = "  " * (level + 1) # Standard indentation for subdirectories

            for file in files:
                tree_lines.append(f"{content_indent}{file}")

        return '\n'.join(tree_lines)
    except OSError as e:
        print(f"Error generating directory tree for {directory_path}: {e}")
        return f"Error generating tree: {e}" # Return error message


def get_file_language(file_path: str) -> str:
    """Generate a string representation of the directory tree.

    This function traverses the specified directory and constructs a visual
    representation of its structure, including subdirectories and files.
    It returns the directory tree as a formatted string.

    Parameters:
    directory_path (str): The path of the directory to traverse.

    Returns:
    str: A formatted string representing the directory tree.

    Exceptions:
    OSError: If an error occurs while accessing the directory."""
    """Determine programming language based on file extension."""
    # Using a simple dictionary for common extensions
    extension_map = {
        '.py': 'Python', '.ipynb': 'Python Notebook',
        '.js': 'JavaScript', '.jsx': 'JavaScript',
        '.ts': 'TypeScript', '.tsx': 'TypeScript',
        '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS',
        '.md': 'Markdown', '.rst': 'reStructuredText',
        '.json': 'JSON', '.xml': 'XML', '.yaml': 'YAML', '.yml': 'YAML',
        '.java': 'Java', '.kt': 'Kotlin', '.scala': 'Scala',
        '.h': 'C/C++', '.hpp': 'C++', '.cpp': 'C++', '.c': 'C',
        '.cs': 'C#', '.go': 'Go', '.php': 'PHP', '.rb': 'Ruby',
        '.rs': 'Rust', '.swift': 'Swift', '.m': 'Objective-C',
        '.r': 'R', '.sh': 'Shell', '.bat': 'Batch', '.ps1': 'PowerShell',
        '.sql': 'SQL', '.dockerfile': 'Dockerfile', 'dockerfile': 'Dockerfile',
        '.tf': 'Terraform', '.hcl': 'HCL',
        '.txt': 'Text', '': 'Unknown' # Handle files with no extension
    }
    # Case-insensitive matching, get extension including '.'
    _, ext = os.path.splitext(file_path)
    language = extension_map.get(ext.lower(), 'Other') # Default to 'Other'

    # Handle files with no extension but common names
    basename = os.path.basename(file_path)
    if not ext and basename.lower() == 'dockerfile':
        language = 'Dockerfile'
    elif not ext and basename.lower() == 'makefile':
        language = 'Makefile'

    return language


def handle_directory_error(directory_path: str) -> bool:
    """Validate directory path exists, is a directory, and is accessible."""
    if not directory_path:
        st.error("Directory path cannot be empty.")
        return False
        """
        Determines the programming language associated with a given file based on its extension.

        Parameters:
        file_path (str): The path of the file whose language is to be identified.

        Returns:
        str: The name of the programming language, or 'Other' if the extension is not recognized.

        Raises:
        None
        """
    if not isinstance(directory_path, str): # Basic type check
        st.error("Directory path must be a string.")
        return False
    try:
        if not os.path.exists(directory_path):
            st.error(f"Directory does not exist: {directory_path}")
            return False
        if not os.path.isdir(directory_path):
            """
            Logs a message to event_log.txt in the specified directory. If the directory does not exist, it will be created.

            Parameters:
            directory_path (str): The path to the directory where the log file will be created.
            message (str): The message to log.

            Returns:
            None

            Raises:
            OSError: If there is an error accessing or creating the log file.
            """
            st.error(f"Path is not a directory: {directory_path}")
            return False
        # Check read permissions
        os.listdir(directory_path)
        return True
    except PermissionError:
        st.error(f"Permission denied accessing directory: {directory_path}")
        return False
    except OSError as e:
        st.error(f"OS error accessing directory {directory_path}: {e}")
        return False
    except Exception as e: # Catch other potential errors
        st.error(f"Unexpected error validating directory {directory_path}: {e}")
        return False


def log_event(directory_path: str, message: str) -> None:
    """Logs a message to event_log.txt in the specified directory."""
    log_filename = "event_log.txt"
    # Define timestamp *before* the try block to ensure it's always bound
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Ensure the directory exists or create it
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        elif not directory_path:
            # Log to current directory if path is invalid/empty
            print(f"Warning: Logging to current directory, invalid path provided: {directory_path}")
            directory_path = "." # Default to current dir

        log_file_path = os.path.join(directory_path, log_filename)

        with open(log_file_path, "a", encoding="utf-8") as f:
            # Use the timestamp defined outside the try block
            f.write(f"{timestamp} - {message}\n")
    except OSError as e:
        print(f"Error writing to log file {log_filename} in {directory_path}: {e}")
        print(f"Original log message: {timestamp} - {message}")
    except Exception as e:
        print(f"Unexpected error during logging: {e}")
        print(f"Original log message: {timestamp} - {message}")


def save_synopsis(directory_path: str, content: str, filename: str) -> bool:
    """Saves content to a file in the specified directory."""
    if not content:
        st.warning("No content provided to save.")
        return False
    if not directory_path or not handle_directory_error(directory_path):
        """Gets description (via summarization) and placeholder use case.

        Parameters:
        file_path (str): Path to the file to be summarized.
        llm_provider (str): The provider of the language model for summarization.

        Returns:
        Tuple[str, str]: A tuple containing the description and use case.

        Raises:
        OSError: If there is an issue accessing the file.
        Exception: For any unexpected errors during processing.
        """
        st.error(f"Invalid directory path for saving: {directory_path}")
        return False

    try:
        file_path = os.path.join(directory_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        # Use st.success cautiously in non-main functions if they might be reused
        # Consider returning success status and path instead
        print(f"Synopsis saved to {file_path}") # Log success
        return True
    except Exception as e:
        st.error(f"Error saving synopsis file {filename}: {e}")
        log_event(directory_path, f"Error saving synopsis file {filename}: {e}")
        return False


# ... inside get_llm_response function ...
def get_llm_response(file_path: str, llm_provider: str) -> Tuple[str, str]:
    """Gets description (via summarization) and placeholder use case."""
    description = "" # Initialize description
    use_case = "N/A" # Initialize use_case

    try:
        # Check file size first to avoid reading huge files
        # This needs to be inside the try block as getsize can fail
        file_size = os.path.getsize(file_path)
        if file_size > 1 * 1024 * 1024: # Limit to 1MB for summarization
            description = "File too large for automatic description."
            # Return early if file is too large, use_case remains "N/A"
            return description, use_case

        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read() # content is defined here

        # Process content only if it exists
        if content:
            # Summarize text
            description = summarize_text(content) # Use content here

            # Determine use case based on filename (basename only)
            basename_lower = os.path.basename(file_path).lower()
            if "test" in basename_lower:
                use_case = "Testing/Verification"
            elif "util" in basename_lower or "helper" in basename_lower:
                use_case = "Utility/Helper Function"
            else:
                """
                Reads the content of a file, summarizes it, and determines its use case based on the filename.

                Parameters:
                - file_path (str): The path to the file to be read.

                Returns:
                - tuple: A description of the content and its use case.

                Exceptions:
                - FileNotFoundError: Raised when the specified file does not exist.
                - OSError: Raised for OS-related errors when accessing the file.
                """
                use_case = "Core Logic/Component"
        else:
            # Handle empty content case
            description = "File is empty."
            # use_case remains "N/A" as initialized

        # Return successfully processed description and use case
        return description, use_case

    except FileNotFoundError:
        # Handle file not found errors specifically
        print(f"File not found: {file_path}") # Log the specific error
        # 'content' is NOT defined here, don't use it
        return "Error: File not found", "Error: File not found"

    except OSError as e:
        # Handle OS-related errors (permissions, getsize fails, etc.)
        print(f"OS error accessing file {file_path}: {e}")
        # 'content' is NOT defined here, don't use it
        return f"Error accessing file: {e}", f"Error: {e}"

    except Exception as e:
        # Handle any other unexpected errors during processing
        print(f"Unexpected error processing file {file_path}: {e}")
        # 'content' is NOT defined here, don't use it
        return f"Error processing file: {e}", f"Error: {e}"


# --- Core Logic Functions ---

def process_repo(
    repo_path: str,
    include_options: Dict[str, bool],
    llm_provider: str
) -> RepoData:
    """Processes a repository path to extract file information."""
    repo_data: RepoData = {
        "repo_path": repo_path,
        "files": [],
        "languages": [],
        "error": None
    }
    # Use handle_directory_error for initial validation
    if not handle_directory_error(repo_path):
        repo_data["error"] = f"Invalid or inaccessible directory: {repo_path}"
        repo_data["languages"] = []
        return repo_data

    try:
        items = traverse_directory(repo_path)
        languages_set = set()

        # Add progress feedback within the loop if many files
        # total_items = len(items)
        # progress_bar = st.progress(0) # Use only if called directly from Streamlit context

        for i, item_path in enumerate(items):
            if os.path.isfile(item_path):
                language = get_file_language(item_path)
                languages_set.add(language)

                file_data: FileData = {"path": item_path, "language": language}

                # --- Token Count ---
                if include_options.get("token_count") and language not in ['Unknown', 'Other']:
                    try:
                        # Check file size before reading
                        if os.path.getsize(item_path) > 5 * 1024 * 1024: # 5MB limit
                            file_data["token_count"] = "File too large to count tokens"
                        else:
                            with open(item_path, "r", encoding="utf-8", errors='ignore') as f:
                                content = f.read()
                            file_data["token_count"] = len(content.split()) # Simple whitespace split
                    except FileNotFoundError:
                        file_data["token_count"] = "File not found during count"
                    except Exception as e:
                        print(f"Error reading for token count {item_path}: {e}")
                        file_data["token_count"] = "Error reading file"

                # --- Descriptions and Use Cases ---
                needs_llm = include_options.get("descriptions") or include_options.get("use_cases")
                if needs_llm and language not in ['Unknown', 'Other']: # Avoid LLM for unknown types
                    description, use_case = get_llm_response(item_path, llm_provider)
                    if include_options.get("descriptions"):
                        file_data["description"] = description
                    if include_options.get("use_cases"):
                        file_data["use_case"] = use_case

                repo_data["files"].append(file_data)
            # Update progress if applicable
            # if total_items > 0: progress_bar.progress((i + 1) / total_items)


        repo_data["languages"] = sorted(list(languages_set))
        return repo_data

    except Exception as e:
        print(f"Unexpected error processing repository {repo_path}: {e}")
        st.error(f"Error processing repository {repo_path}: {e}") # Keep st error here as it's a major failure point
        return RepoData(
            repo_path=repo_path,
            files=[],
            error=f"Unexpected processing error: {str(e)}",
            languages=None
        )


def generate_synopsis_text(
    repo_data: RepoData,
    include_tree: bool,
    directory_path: str # Need original path for tree generation
) -> str:
    """Generates the synopsis text content from processed RepoData."""
    synopsis_parts = []
    languages = repo_data.get("languages")
    files = repo_data.get("files", [])

    if languages:
        synopsis_parts.append(f"Languages used: {', '.join(languages)}\n")

    if include_tree:
        synopsis_parts.append("## Directory Tree")
        tree = generate_directory_tree(directory_path)
        synopsis_parts.append(tree)

    if files:
        synopsis_parts.append("\n## File Details")
        for file_data in files:
            path = file_data.get('path', 'N/A')
            lang = file_data.get('language', 'N/A')
            synopsis_parts.append(f"\n### File: `{path}` ({lang})") # Use markdown code format

            details = []
            if 'token_count' in file_data:
                details.append(f"- **Token Count:** {file_data['token_count']}")
            if 'description' in file_data:
                 details.append(f"- **Description:** {file_data['description']}")
            if 'use_case' in file_data:
                 details.append(f"- **Use Case:** {file_data['use_case']}")

            if details:
                synopsis_parts.extend(details) # Add details as separate list items


    return "\n".join(synopsis_parts)


# --- Main Application Logic ---

def main():
    st.set_page_config(layout="wide") # Use wider layout
    st.title("Repository Synopsis Generator")

    # --- Configuration Sidebar ---
    st.sidebar.header("Configuration")
    base_directory = st.sidebar.text_input("Base Directory Containing Repositories:", key="base_dir")

    st.sidebar.subheader("Synopsis Options")
    include_tree = st.sidebar.checkbox("Include Directory Tree", value=True, key="inc_tree")
    include_descriptions = st.sidebar.checkbox("Include Descriptions (Summaries)", value=True, key="inc_desc")
    include_token_count = st.sidebar.checkbox("Include Token Count", value=True, key="inc_token")
    include_use_cases = st.sidebar.checkbox("Include Use Cases (Placeholder)", value=True, key="inc_use")
    llm_provider = st.sidebar.selectbox(
        "LLM Provider (Placeholder)",
        ("Groq", "Other"), # Simplified
        key="llm_select",
        help="LLM selection currently affects summarization only."
    )

    # --- Repository Selection Area ---
    selected_repo_paths_relative = []
    repo_names = []

    st.header("Repository Selection")
    if base_directory and handle_directory_error(base_directory):
        try:
            repo_names = sorted([
                name for name in os.listdir(base_directory)
                if os.path.isdir(os.path.join(base_directory, name))
            ])

            if not repo_names:
                st.warning(f"No subdirectories found in {base_directory}.")
            else:
                # Use columns for better layout if many repos
                col1, col2 = st.columns([3,1]) # Adjust ratio as needed
                with col1:
                    selected_repo_paths_relative = st.multiselect(
                        "Select repositories to process:",
                        repo_names,
                        default=repo_names,
                        key="repo_select"
                    )
                with col2:
                    # Add Select All / Deselect All buttons
                    if st.button("Select All", key="select_all"):
                        # Need to use session state to update multiselect properly
                        st.session_state.repo_select = repo_names
                        st.rerun() # Rerun to reflect changes
                    if st.button("Deselect All", key="deselect_all"):
                        st.session_state.repo_select = []
                        st.rerun()

        except Exception as e:
            st.error(f"Error listing repositories in {base_directory}: {e}")
    elif base_directory:
        st.info("Enter a valid and accessible base directory to see repositories.")
    else:
        st.info("Enter a base directory path in the sidebar.")


    # --- Action Button and Processing ---
    st.header("Generate Synopsis")
    if st.button("Generate", key="generate_button", type="primary"):
        # --- Input Validation ---
        if not base_directory or not handle_directory_error(base_directory):
            st.error("Please enter a valid base directory in the sidebar.")
            return
        if 'repo_select' not in st.session_state or not st.session_state.repo_select:
            st.warning("Please select at least one repository.")
            return

        selected_repo_paths_relative = st.session_state.repo_select
        selected_full_paths = [os.path.join(base_directory, name) for name in selected_repo_paths_relative]

        include_options = {
            "descriptions": include_descriptions,
            "token_count": include_token_count,
            "use_cases": include_use_cases,
            # 'tree' option is handled separately during text generation
        }

        # --- Processing Loop ---
        all_repo_data: Dict[str, RepoData] = {}
        combined_synopsis_text = ""
        has_errors = False
        start_time = datetime.datetime.now()

        st.info(f"Starting processing for {len(selected_full_paths)} repositories...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, repo_path in enumerate(selected_full_paths):
            repo_name = os.path.basename(repo_path)
            status_text.text(f"Processing: {repo_name} ({i+1}/{len(selected_full_paths)})...")

            result_data = process_repo(repo_path, include_options, llm_provider)
            all_repo_data[repo_path] = result_data # Store raw data

            if result_data.get("error"):
                has_errors = True
                st.warning(f"Finished {repo_name} with error: {result_data['error']}")
                log_event(base_directory, f"ERROR processing {repo_name}: {result_data['error']}")
            else:
                # Generate synopsis text for this repo and append
                repo_synopsis = generate_synopsis_text(result_data, include_tree, repo_path)
                combined_synopsis_text += f"\n\n---\n\n## Repository: {repo_name}\n\n{repo_synopsis}"
                log_event(base_directory, f"Successfully processed {repo_name}")

            progress_bar.progress((i + 1) / len(selected_full_paths))

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        status_text.text(f"Processing finished in {duration.total_seconds():.2f} seconds.")

        # --- Display and Save Results ---
        st.header("Results")
        if has_errors:
            st.error("Some repositories encountered errors. Check logs and output files.")

        if combined_synopsis_text:
            # Display combined synopsis in an expander
            with st.expander("View Combined Synopsis Text", expanded=False):
                st.markdown(combined_synopsis_text) # Use markdown for better formatting

            # Save combined synopsis text file
            txt_filename = f"combined_synopsis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md" # Save as markdown
            if save_synopsis(base_directory, combined_synopsis_text, txt_filename):
                st.success(f"Combined synopsis saved to {os.path.join(base_directory, txt_filename)}")
                # Add download button for the text file
                try:
                    with open(os.path.join(base_directory, txt_filename), "r", encoding="utf-8") as fp:
                        st.download_button(
                                label="Download Synopsis (.md)",
                                data=fp,
                                file_name=txt_filename,
                                mime="text/markdown"
                        )
                except Exception as e:
                    st.warning(f"Could not create download button for synopsis text: {e}")

        # Save detailed JSON data
        json_filename = f"repo_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_filepath = os.path.join(base_directory, json_filename)
        try:
            with open(json_filepath, "w", encoding='utf-8') as f:
                json.dump(all_repo_data, f, indent=4, ensure_ascii=False)
            st.success(f"Detailed JSON data saved to {json_filepath}")
            log_event(base_directory, f"Detailed JSON data saved to {json_filepath}")
            # Add download button for the JSON file
            with open(json_filepath, "rb") as fp: # Read as bytes for download button
                st.download_button(
                    label="Download Detailed JSON Data",
                    data=fp,
                    file_name=json_filename,
                    mime="application/json"
                )
        except Exception as e:
            st.error(f"Error saving detailed JSON data: {e}")
            log_event(base_directory, f"Error saving detailed JSON data: {e}")


if __name__ == "__main__":
    main()
