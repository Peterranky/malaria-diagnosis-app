import os

def create_claude_response():
    base_dir = os.path.dirname(__file__)
    output_file = os.path.join(base_dir, 'response_to_claude.md')
    
    # Files to include
    files_to_read = {
        'ui/auth_ui.py': os.path.join(base_dir, 'ui', 'auth_ui.py'),
        'ui/diagnosis_ui.py': os.path.join(base_dir, 'ui', 'diagnosis_ui.py'),
        'ui/risk_ui.py': os.path.join(base_dir, 'ui', 'risk_ui.py'),
        'ui/combined_ui.py': os.path.join(base_dir, 'ui', 'combined_ui.py'),
        'src/risk_pipeline/inference.py': os.path.join(base_dir, 'src', 'risk_pipeline', 'inference.py'),
        'src/image_pipeline/train.py (Snippet - Metrics Fix)': os.path.join(base_dir, 'src', 'image_pipeline', 'train.py')
    }
    
    # Generate tree
    tree_lines = []
    for root, dirs, files in os.walk(base_dir):
        # Exclude hidden folders and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree_lines.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if not f.startswith('.'):
                tree_lines.append(f"{subindent}{f}")
    tree_str = "\n".join(tree_lines)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("# Response to Claude (Project Documentation)\n\n")
        
        # 1. Answer to the auth question
        out.write("## 1. Authentication System Confirmation\n")
        out.write("Regarding your question on the authentication mechanism:\n")
        out.write("**It is a mock stub.** The application writes to the SQLite `users` table, but passwords are currently stored in plain text. The column is named `password_hash` to reflect a production-ready schema design, but for this prototype, `app.py` explicitly notes that this is 'mock auth'. To claim cryptographic security in the documentation would be overclaiming.\n\n")
        
        # 2. Directory Tree
        out.write("## 2. Project Directory Tree\n")
        out.write("```text\n")
        out.write(tree_str)
        out.write("\n```\n\n")
        
        # 3. Source Files
        out.write("## 3. Source Code Files (Appendix A)\n\n")
        
        for name, path in files_to_read.items():
            out.write(f"### {name}\n")
            out.write("```python\n")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if name == 'src/image_pipeline/train.py (Snippet - Metrics Fix)':
                        # Just grab the metrics calculation block to prove it was fixed
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'cm = confusion_matrix' in line:
                                start = i - 2
                                end = i + 20
                                out.write("\n".join(lines[start:end]) + "\n")
                                break
                    else:
                        out.write(content)
            except Exception as e:
                out.write(f"# Error reading file: {e}")
            out.write("\n```\n\n")

    print(f"Generated {output_file}")

if __name__ == "__main__":
    create_claude_response()
