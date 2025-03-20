import os
import unittest
import tempfile
import shutil
import zipfile
import csv
from datetime import datetime
from unittest.mock import MagicMock, patch

# Import your modules
from scrape import NCERTDownloader
from text_ext import NCERTPDFExtractor
import app

#############################################
# Tests for scraper.py (NCERTDownloader)
#############################################


class TestNCERTDownloader(unittest.TestCase):
    def setUp(self):
        self.download_dir = tempfile.mkdtemp()
        self.extract_dir = tempfile.mkdtemp()
        self.patcher = patch.object(
            NCERTDownloader, "initialize_driver", return_value=MagicMock()
        )
        self.mock_init_driver = self.patcher.start()
        self.downloader = NCERTDownloader(
            "dummy_geckodriver_path", self.download_dir, self.extract_dir
        )

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.download_dir)
        shutil.rmtree(self.extract_dir)

    def test_rename_latest_zip(self):
        dummy_zip_path = os.path.join(self.download_dir, "dummy.zip")
        with open(dummy_zip_path, "wb") as f:
            f.write(b"Dummy content")
        os.utime(
            dummy_zip_path, (datetime.now().timestamp(), datetime.now().timestamp())
        )

        new_path = self.downloader.rename_latest_zip(7)
        expected_filename = "Grade7.zip"
        self.assertTrue(
            os.path.exists(os.path.join(self.download_dir, expected_filename))
        )
        self.assertEqual(new_path, os.path.join(self.download_dir, expected_filename))

    def test_unzip_for_grade(self):
        grade = 8
        dummy_zip_path = os.path.join(self.download_dir, "dummy.zip")
        dummy_txt_name = "test.txt"
        dummy_txt_content = "Hello, NCERT!"
        with zipfile.ZipFile(dummy_zip_path, "w") as zf:
            zf.writestr(dummy_txt_name, dummy_txt_content)

        # Unzip the file and check if the content is extracted properly
        self.downloader.unzip_for_grade(grade, dummy_zip_path)
        grade_folder = os.path.join(self.extract_dir, f"Grade{grade}")
        extracted_file = os.path.join(grade_folder, dummy_txt_name)
        self.assertTrue(os.path.exists(extracted_file))
        with open(extracted_file, "r") as f:
            content = f.read()
        self.assertEqual(content, dummy_txt_content)


#############################################
# Tests for pdf_text_txt.py (NCERTPDFExtractor)
#############################################


class TestNCERTPDFExtractor(unittest.TestCase):
    def setUp(self):
        self.extracted_dir = tempfile.mkdtemp()
        self.output_json = os.path.join(self.extracted_dir, "output.json")
        self.grade = 9
        self.grade_folder = os.path.join(self.extracted_dir, f"Grade{self.grade}")
        os.makedirs(self.grade_folder, exist_ok=True)

        from PyPDF2 import PdfWriter

        self.dummy_pdf_path = os.path.join(self.grade_folder, "dummy.pdf")
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(self.dummy_pdf_path, "wb") as f:
            writer.write(f)

    def tearDown(self):
        shutil.rmtree(self.extracted_dir)

    def test_extract_text_from_pdf(self):
        extractor = NCERTPDFExtractor(
            self.extracted_dir, self.output_json, [self.grade]
        )
        text = extractor.extract_text_from_pdf(self.dummy_pdf_path)
        self.assertEqual(text, "")

    def test_build_json(self):
        extractor = NCERTPDFExtractor(
            self.extracted_dir, self.output_json, [self.grade]
        )
        data = extractor.build_json()
        self.assertIn(f"Grade {self.grade}", data)
        # Check that our dummy.pdf is present
        grade_entries = data[f"Grade {self.grade}"]
        self.assertTrue(
            any(entry["filename"] == "dummy.pdf" for entry in grade_entries)
        )


#############################################
# Tests for app.py (Helper Functions)
#############################################


class TestAppHelpers(unittest.TestCase):
    def test_clean_text(self):
        text = "   This   is   a   test   "
        cleaned = app.clean_text(text)
        self.assertEqual(cleaned, "This is a test")

    def test_construct_prompt(self):
        context = "Context info"
        question = "What is testing?"
        fake_history = [("Q1", "A1"), ("Q2", "A2")]
        with patch.dict(app.st.session_state, {"history": fake_history}, clear=True):
            prompt = app.construct_prompt(context, question)
            self.assertIn("Context info", prompt)
            self.assertIn("What is testing?", prompt)
            self.assertIn("Conversation History:", prompt)

    def test_log_conversation_csv(self):
        with tempfile.NamedTemporaryFile("r+", delete=False) as tmpfile:
            logfile = tmpfile.name
        question = "Test question"
        answer = "Test answer"
        app.log_conversation_csv(question, answer, logfile=logfile)
        with open(logfile, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        os.remove(logfile)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[-1]["question"], question)
        self.assertEqual(rows[-1]["answer"], answer)


#############################################
# Run all tests
#############################################

if __name__ == "__main__":
    unittest.main()
