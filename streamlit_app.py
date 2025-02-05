import streamlit as st
import os
import datetime

st.title("Repo Synopsis Generator")

st.subheader("Synopsis Options")
include_tree = st.checkbox("Directory Tree", value=True)
include_descriptions = st.checkbox("Descriptions", value=True)
include_token_count = st.checkbox("Token Count", value=True)
include_use_cases = st.checkbox("Use Cases", value=True)

# API keys should be stored in the user's environmental variables
llm_provider = st.selectbox(
    "Select LLM API Provider",
    ("Groq", "Cerberas", "SombaNova", "Gemini"),
)

directory_path = st.text_input("Enter directory path:")

def traverse_directory(directory_path):
    items = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        items.append(item_path)
        if os.path.isdir(item_path):
            items.extend(traverse_directory(item_path))
    return items

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

def log_event(directory_path, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_path = os.path.join(directory_path, "event_log.txt") if directory_path else "event_log.txt"
    try:
        with open(log_file_path, "a") as f:
            f.write(f"{timestamp} - {message}\\n")
    except Exception as e:
        st.write(f"Error writing to log file: {e}")

def save_synopsis(directory_path, synopsis):
    if not directory_path:
        st.write("Please enter a directory path to save the synopsis.")
        return

    file_path = os.path.join(directory_path, "repo_synopsis.md")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(synopsis)
        st.write(f"Synopsis saved to {file_path}")
        log_event(directory_path, f"Synopsis saved to {file_path}")
    except Exception as e:
        st.write(f"Error saving synopsis: {e}")
        log_event(directory_path, f"Error saving synopsis: {e}")

def generate_synopsis(directory_path, include_tree, include_descriptions, include_token_count, include_use_cases, llm_provider):
    st.write(f"Generating synopsis for: {directory_path}")

    if not directory_path:
        st.write("Please enter a directory path.")
        return

    items = traverse_directory(directory_path)
    st.write(f"Found {len(items)} items in the directory.")

    synopsis = ""
    languages = set()

    if include_tree:
        synopsis += "## Directory Tree\\n"
        synopsis += generate_directory_tree(directory_path).replace('\\\\n', '\\n') + "\\n"

    if include_descriptions or include_token_count or include_use_cases:
        synopsis += "## Item Details\\n"
        for item_path in items:
            if os.path.isfile(item_path):
                language = get_file_language(item_path)
                languages.add(language)
                if include_token_count and language != "Unknown":
                    try:
                        with open(item_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            token_count = len(content.split())
                            synopsis += f"- File: {item_path}, Language: {language}, Token Count: {token_count}\\n"
                    except Exception as e:
                        synopsis += f"- Error reading file {item_path}: {e}\\n"
                elif include_descriptions:
                    synopsis += f"- File: {item_path}, Language: {language}\\n"
                if include_descriptions or include_use_cases:
                    description, use_case = get_llm_response(item_path, llm_provider)
                    if include_descriptions:
                        synopsis += f"  - Description: {description}\\n"
                    if include_use_cases:
                        synopsis += f"  - Use Case: {use_case}\\n"
            elif include_descriptions:
                synopsis += f"- Directory: {item_path}\\n"
                if include_descriptions or include_use_cases:
                    description, use_case = get_llm_response(item_path, llm_provider)
                    if include_descriptions:
                        synopsis += f"  - Description: {description}\\n"
                    if include_use_cases:
                        synopsis += f"  - Use Case: {use_case}\\n"

    if languages:
        synopsis = f"Languages used: {', '.join(languages)}\\n\\n" + synopsis

    st.subheader("Synopsis Preview")
    synopsis_review = st.text_area("Synopsis", value=synopsis, height=300)
    log_event(directory_path, "Synopsis generated")

    if st.button("Save Synopsis"):
        save_synopsis(directory_path, synopsis_review)

    # Removing this button as it was causing the function to be called twice and the synopsis to be generated twice
    # with st.spinner("Generating synopsis..."):
    #     if st.button("Generate Synopsis"):
    #         generate_synopsis(directory_path, include_tree, include_descriptions, include_token_count, include_use_cases, llm_provider)

if st.button("Proceed"):
    def generate_synopsis_wrapper():
        directory_path_value = directory_path.strip().replace("\\", "\\\\")  # ADD .replace()
        print(f"Directory Path Value: {directory_path_value}")
        generate_synopsis(directory_path_value, include_tree, include_descriptions, include_token_count, include_use_cases, llm_provider)
    generate_synopsis_wrapper()