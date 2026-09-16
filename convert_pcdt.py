import pdfplumber
from pathlib import Path
from typing import Union

def convert_pdf_to_markdown(pdf_path: Union[str, Path], output_path: Union[str, Path]):
    pdf_file = Path(pdf_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    markdown_lines = []
    
    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        print(f"Iniciando a extração de {total_pages} páginas...")
        
        for i, page in enumerate(pdf.pages):
            markdown_lines.append(f"\n\n<!-- Página {i + 1} -->\n")
            
            text = page.extract_text(layout=False)
            if text:
                markdown_lines.append(text)
            
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    markdown_lines.append("\n")
                    for row_idx, row in enumerate(table):
                        cleaned_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
                        markdown_lines.append("| " + " | ".join(cleaned_row) + " |")
                        if row_idx == 0:
                            markdown_lines.append("|" + "|".join(["---"] * len(row)) + "|")
                    markdown_lines.append("\n")

    out_file.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Sucesso! Arquivo Markdown gerado em: {out_file}")

if __name__ == "__main__":
    convert_pdf_to_markdown(
        pdf_path="diretivas-saude.pdf",
        output_path="protocols/pcdt_cancer_mama_2024.md"
    )