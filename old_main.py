import pymupdf  # PyMuPDF
import json
import os
from transformers import pipeline

# -------- PDF Section --------
def insert_section(section_stack, level, title, content):
    new_section = {"_content": content}
    section_stack = section_stack[:level]
    parent = section_stack[-1] if section_stack else section_tree
    parent[title] = new_section
    section_stack.append(new_section)
    return section_stack

def extract_sections_from_toc(pdf_path):
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc(simple=True)
    if not toc:
        print(f"No Table of Contents found in {pdf_path}")
        return {}

    global section_tree
    section_tree = {}
    section_stack = [section_tree]

    for i, (level, title, page_num) in enumerate(toc):
        title = title.strip()
        start_page = page_num - 1
        end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc)
        if start_page == end_page:
            end_page += 1
        content = "".join(doc[p].get_text() for p in range(start_page, end_page))
        section_stack = insert_section(section_stack, level, title, content.strip())

    return section_tree

def process_all_pdfs(input_folder_process, output_folder_process):
    os.makedirs(output_folder_process, exist_ok=True)
    for filename in os.listdir(input_folder_process):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder_process, filename)
            print(f"[PDF] Processing {filename}...")
            structured = extract_sections_from_toc(pdf_path)
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_folder_process, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, indent=2, ensure_ascii=False)
            print(f"[PDF] Saved: {json_path}")

# -------- ClinicalTrials.gov JSON Section --------
def flatten_json_to_sections(data, parent_path=[]):
    result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full_path = parent_path + [key]
            if isinstance(value, (dict, list)):
                result.update(flatten_json_to_sections(value, full_path))
            else:
                key_str = " / ".join(full_path)
                result[key_str] = str(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            full_path = parent_path + [f"[{i}]"]
            if isinstance(item, (dict, list)):
                result.update(flatten_json_to_sections(item, full_path))
            else:
                key_str = " / ".join(full_path)
                result[key_str] = str(item)
    return result

def nest_flattened_sections(flat_sections):
    nested = {}
    for path_str, content in flat_sections.items():
        keys = path_str.split(" / ")
        current = nested
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = {"_content": content}
    return nested

def process_all_json_trials(input_folder_process, output_folder_process):
    os.makedirs(output_folder_process, exist_ok=True)
    for filename in os.listdir(input_folder_process):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(input_folder_process, filename)
            print(f"[JSON] Processing {filename}...")
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    raw_json = json.load(f)
                except json.JSONDecodeError:
                    print(f"[JSON] Skipped: {filename} (Invalid JSON)")
                    continue
            flat = flatten_json_to_sections(raw_json)
            nested = nest_flattened_sections(flat)
            output_filename = os.path.splitext(filename)[0] + "_structured.json"
            output_path = os.path.join(output_folder_process, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(nested, f, indent=2, ensure_ascii=False)
            print(f"[JSON] Saved: {output_path}")

# -------- QA Pipeline --------
def load_all_documents(output_folder):
    documents = {}
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(".json"):
            with open(os.path.join(output_folder, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                documents[filename] = data
    return documents

def extract_text_from_nested_json(data):
    texts = []
    def recurse(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "_content":
                    texts.append(v)
                else:
                    recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    recurse(data)
    return "\n".join(texts)

def ask_question(question, context):
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return qa(question=question, context=context)

def interactive_qa(documents):
    combined_context = "\n".join(extract_text_from_nested_json(doc) for doc in documents.values())
    print("\nEnter your question (or 'exit' to quit):")
    while True:
        q = input("Q: ")
        if q.lower() == "exit":
            break
        result = ask_question(q, combined_context)
        print(f"A: {result['answer']} (score: {result['score']:.2f})\n")

# -------- Runner --------
if __name__ == "__main__":
    base_dir = os.getcwd()
    input_pdf_dir = os.path.join(base_dir, "reports")
    input_json_dir = os.path.join(base_dir, "json_reports")
    output_dir = os.path.join(base_dir, "output")

    process_all_pdfs(input_pdf_dir, output_dir)
    process_all_json_trials(input_json_dir, output_dir)

    documents = load_all_documents(output_dir)
    interactive_qa(documents)
