import re

def process_tex_content(content):
    """
    Processes the content of a TeX file. For each tcolorbox environment:
      - If it already contains a caption (\captionof{figure}), leave it unchanged.
      - Otherwise, if the environment contains Python code (detected via \begin{lstlisting}),
        do not add any caption.
      - Otherwise, add an empty caption (\captionof{figure}{}).
    """
    # Regular expression to match tcolorbox environments with options.
    pattern = r"(\\begin\{tcolorbox\}\[.*?\])(.*?)(\\end\{tcolorbox\})"
    
    def add_caption(match):
        begin_box = match.group(1)
        body = match.group(2)
        end_box = match.group(3)
        # If a caption is already present, leave the environment unchanged.
        if r"\captionof{figure}" in body:
            return match.group(0)
        # If the box contains Python code (lstlisting), do not add a caption.
        if r"\begin{lstlisting}" in body:
            return match.group(0)
        # Otherwise, insert an empty caption command.
        new_body = body + "\n\\captionof{figure}{}"
        return begin_box + new_body + end_box

    # Use re.DOTALL to ensure newlines are included.
    return re.sub(pattern, add_caption, content, flags=re.DOTALL)

# Prompt user for file names without using sys.argv or any parsing library.
input_filename = input("Enter the input TeX filename: ")
output_filename = input("Enter the output TeX filename: ")

# Read the entire TeX file content.
with open(input_filename, "r", encoding="utf-8") as file:
    tex_content = file.read()

# Process the content.
modified_content = process_tex_content(tex_content)

# Write the modified content to the output file.
with open(output_filename, "w", encoding="utf-8") as file:
    file.write(modified_content)

print("Processing complete. Modified file saved as", output_filename)
