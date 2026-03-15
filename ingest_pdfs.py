import fitz  # This is the PyMuPDF library
import os

def process_pdfs(root_folder):
    extracted_data = []

    # This loop 'walks' through every subfolder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                
                # We'll use the subfolder name as the 'Category'
                category = os.path.basename(root)
                
                print(f"Reading: {category} > {filename}...")
                
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text()
                        
                        if text.strip():
                            extracted_data.append({
                                "source": f"{category} / {filename}",
                                "page": page_num + 1,
                                "content": text.strip()
                            })
                    doc.close()
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    return extracted_data

# The path to your specific knowledge base
knowledge_path = r"C:\Users\Home\Downloads\My AI Project\knowledge_source"
data = process_pdfs(knowledge_path)

print(f"\nSuccess! Extracted {len(data)} pages from your PDFs.")