import re

def find_missing_index(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    valid_envs = re.compile(r'\\begin{h?(theorem|corollary|proposition|lemma|definition|notation|remark|conjecture|axiom|example|examples|exercise|exercises|counterexample|construction|warning|digression|perspective|discussion|terminology|heuristics)}', re.IGNORECASE)
    
    for i, line in enumerate(lines):
        if valid_envs.search(line) and not re.search(r'\\index{.*?}', line):
            prev_line = lines[i - 1].strip() if i > 0 else "(No previous line)"
            next_line = lines[i + 1].strip() if i < len(lines) - 1 else "(No next line)"
            print(f"> {prev_line}")
            print(f"L{i+1}: {line.strip()} < no '\index' found")
            print(f"> {next_line}\n")

# Example usage:
filename = "cap/01-categorie.tex"  # Replace with the actual filename
find_missing_index(filename)