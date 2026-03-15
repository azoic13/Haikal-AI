import fitz  # PyMuPDF
import os

def process_all_folders(root_path):
    knowledge_base = []

    # os.walk travels through the main folder and all subfolders
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            if filename.endswith(".pdf"):
                # This gets the full path to the file
                file_path = os.path.join(root, filename)
                
                # We can use the folder name as part of the "Source"
                # This helps the AI know the category (e.g., 'Rituals')
                category = os.path.basename(root) 
                
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text()
                        
                        if text.strip(): # Only add if there is actual text
                            knowledge_base.append({
                                "source": f"{category} > {filename}",
                                "page": page_num + 1,
                                "content": text.strip()
                            })
                    doc.close()
                except Exception as e:
                    print(f"Could not read {filename}: {e}")

    return knowledge_base

# Your specific path
path = r"C:\Users\Home\Downloads\My AI Project\knowledge_source"
all_data = process_all_folders(path)

print(f"Total pages processed: {len(all_data)}")
# Let's look at one example
if all_data:
    print(f"Sample Source: {all_data[0]['source']}")