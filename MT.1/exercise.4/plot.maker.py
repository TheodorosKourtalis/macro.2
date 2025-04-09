import os
from googletrans import Translator, LANGUAGES
from googletrans.constants import DEFAULT_USER_AGENT
import httpx  # Χρησιμοποιούμε httpx για να εντοπίσουμε το Timeout

# Ο φάκελος που περιέχει τα .png αρχεία
folder = "/Users/thodoreskourtales/TK.MT.1/exercise.4"

# Βρες όλα τα αρχεία που τελειώνουν σε .png, ταξινομημένα για προβλέψιμη σειρά
png_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])

# Δημιουργία αντικειμένου μεταφραστή
translator = Translator()

# Όνομα αρχείου LaTeX που θα δημιουργηθεί
output_tex = "all_plots_boxes.tex"

with open(output_tex, "w", encoding="utf-8") as texfile:
    texfile.write("% Αυτόματα παραγόμενο αρχείο με όλα τα plots\n")
    texfile.write("\\chapter{Παράρτημα: Παραγόμενα Διαγράμματα}\n\n")
    texfile.write("\\graphicspath{{" + folder + "/}}\n\n")
    
    for png in png_files:
        base_name = os.path.splitext(png)[0]
        base_name_for_translation = base_name.replace("_", " ")
        
        # Προσπάθεια μετάφρασης με try/except για να αποφύγουμε το timeout
        try:
            translated_title = translator.translate(base_name_for_translation, src='en', dest='el').text
        except Exception as e:
            print(f"Μετάφραση απέτυχε για '{base_name_for_translation}': {e}")
            # Χρησιμοποιούμε ως fallback το αρχικό όνομα
            translated_title = base_name_for_translation
        
        # Δημιουργούμε το tcolorbox με τον μεταφρασμένο τίτλο
        texfile.write("\\begin{tcolorbox}[colback=white,colframe=black,title={" + translated_title + "}]\n")
        texfile.write("  \\centering\n")
        texfile.write("  \\includegraphics[width=0.8\\textwidth]{" + png + "}\n")
        texfile.write("  \\vspace{0.5em}\n")
        texfile.write("\\end{tcolorbox}\n\n")
        texfile.write("\\FloatBarrier\n\n")

print(f"Το αρχείο '{output_tex}' δημιουργήθηκε επιτυχώς!")