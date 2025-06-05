import pymupdf
import json
import os


def insert_section(section_stack, level, title, content):
    """Inserts a section at the correct level in the nested dictionary."""
    new_section = {"_content": content}
    # Trim stack to current level
    section_stack = section_stack[:level]
    # Insert into correct parent
    parent = section_stack[-1] if section_stack else section_tree
    parent[title] = new_section
    section_stack.append(new_section)
    return section_stack


def extract_sections_from_toc(pdf_path):
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc(simple=True)  # (level, title, page_num)

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

        content = ""
        for p in range(start_page, end_page):
            content += doc[p].get_text()

        section_stack = insert_section(section_stack, level, title, content.strip())

    return section_tree


def process_all_pdfs(input_folder_process, output_folder_process):
    os.makedirs(output_folder_process, exist_ok=True)

    for filename in os.listdir(input_folder_process):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder_process, filename)
            print(f"Processing {filename}...")

            structured = extract_sections_from_toc(pdf_path)

            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_folder_process, json_filename)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, indent=2, ensure_ascii=False)

            print(f"Saved: {json_path}")


if __name__ == "__main__":
    input_dir = os.path.join(os.getcwd(), "reports")
    output_dir = os.path.join(os.getcwd(), "output")

    process_all_pdfs(input_dir, output_dir)
