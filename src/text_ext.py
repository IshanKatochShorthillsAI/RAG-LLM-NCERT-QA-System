import os
import json
import logging
import PyPDF2
from typing import List, Dict

# -------------------------
# Configuration and Logging
# -------------------------
logging.basicConfig(level=logging.INFO)


class NCERTPDFExtractor:
    def __init__(self, extracted_dir: str, output_json: str, grades: List[int]):
        """
        Initialize with the directory containing grade-specific folders,
        the output JSON file path, and a list of grades to process.
        """
        self.extracted_dir = extracted_dir
        self.output_json = output_json
        self.grades = grades

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyPDF2.
        """
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
        return text.strip()

    def process_grade(self, grade: int) -> List[Dict[str, str]]:
        """
        Process all PDF files in the folder for a given grade.
        Returns a list of dictionaries each with keys 'filename' and 'text'.
        """
        grade_folder = os.path.join(self.extracted_dir, f"Grade{grade}")
        pdf_entries = []
        if not os.path.exists(grade_folder):
            logging.warning(
                f"Folder for Grade {grade} not found in {self.extracted_dir}."
            )
            return pdf_entries

        for filename in os.listdir(grade_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(grade_folder, filename)
                logging.info(f"Extracting text from {pdf_path}")
                text = self.extract_text_from_pdf(pdf_path)
                pdf_entries.append({"filename": filename, "text": text})
        return pdf_entries

    def build_json(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Build a JSON dictionary segregating PDFs by grade.
        """
        data = {}
        for grade in self.grades:
            logging.info(f"Processing Grade {grade}")
            data[f"Grade {grade}"] = self.process_grade(grade)
        return data

    def save_json(self, data: Dict[str, List[Dict[str, str]]]):
        """
        Save the JSON dictionary to the output file.
        """
        try:
            with open(self.output_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logging.info(f"Saved extracted data to {self.output_json}")
        except Exception as e:
            logging.error(f"Error saving JSON to {self.output_json}: {e}")


def main():
    # Update these paths as needed:
    extracted_dir = "/home/shtlp_0133/Documents/RAG Assignment 17 March/Final Folder/Extracted"  # Folder with grade subfolders
    output_json = "/home/shtlp_0133/Documents/RAG Assignment 17 March/Final Folder/grades_text.json"
    grades = [7, 8, 9, 10]

    extractor = NCERTPDFExtractor(extracted_dir, output_json, grades)
    data = extractor.build_json()
    extractor.save_json(data)


if __name__ == "__main__":
    main()
