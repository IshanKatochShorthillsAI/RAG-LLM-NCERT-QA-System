import os
import time
import logging
import zipfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service


class NCERTDownloader:
    def __init__(
        self,
        geckodriver_path: str,
        download_dir: str,
        extract_dir: str,
        url: str = "https://ncert.nic.in/textbook.php",
    ):
        """
        Initialize the downloader with the geckodriver path, download folder, extraction folder, and base URL.
        """
        self.url = url
        self.geckodriver_path = geckodriver_path
        self.download_dir = download_dir
        self.extract_dir = extract_dir
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)
        self.driver = self.initialize_driver()

    def initialize_driver(self) -> webdriver.Firefox:
        """
        Initialize and return a Firefox WebDriver with custom download preferences.
        """
        serv_obj = Service(self.geckodriver_path)
        firefox_options = webdriver.FirefoxOptions()
        # Configure Firefox to use the custom download folder
        firefox_options.set_preference("browser.download.folderList", 2)
        firefox_options.set_preference("browser.download.dir", self.download_dir)
        # Automatically download PDFs and ZIP files without prompt
        firefox_options.set_preference(
            "browser.helperApps.neverAsk.saveToDisk", "application/pdf, application/zip"
        )
        # Disable the built-in PDF viewer
        firefox_options.set_preference("pdfjs.disabled", True)
        driver = webdriver.Firefox(service=serv_obj, options=firefox_options)
        driver.implicitly_wait(10)
        return driver

    def rename_latest_zip(self, grade: int) -> str:
        """
        Wait for a zip file to appear in the download folder and rename it to a grade-specific name.
        Returns the new path of the zip file.
        """
        wait_time = 0
        zip_files = []
        # Wait up to 30 seconds for a zip file to appear
        while wait_time < 30:
            zip_files = [f for f in os.listdir(self.download_dir) if f.endswith(".zip")]
            if zip_files:
                break
            time.sleep(2)
            wait_time += 2

        if not zip_files:
            print(f"No zip file found for Grade {grade} after waiting.")
            return None

        # Pick the newest zip file based on modification time
        latest_file = max(
            zip_files,
            key=lambda f: os.path.getmtime(os.path.join(self.download_dir, f)),
        )
        old_path = os.path.join(self.download_dir, latest_file)
        new_filename = f"Grade{grade}.zip"
        new_path = os.path.join(self.download_dir, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {latest_file} to {new_filename}")
        return new_path

    def unzip_for_grade(self, grade: int, zip_path: str):
        """
        Extract the given zip file into a grade-specific folder inside the extraction directory.
        """
        if not zip_path or not os.path.exists(zip_path):
            print(f"Zip file for Grade {grade} not found.")
            return
        grade_folder = os.path.join(self.extract_dir, f"Grade{grade}")
        os.makedirs(grade_folder, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(grade_folder)
            print(f"Extracted {zip_path} into {grade_folder}")
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")

    def download_for_class(self, class_num: int):
        """
        Process download for a given class number:
          - Navigate to the page, select the proper options, and click "Go".
          - Click the download link.
          - Wait for at least 60 seconds.
          - Rename the downloaded zip file to "Grade{class_num}.zip".
          - Unzip the file into a grade-specific folder.
          - Return to the original URL.
        """
        driver = self.driver
        driver.get(self.url)
        print(f"Processing Class {class_num}...")

        # Select class, subject, and book
        class_dropdown = Select(driver.find_element(By.NAME, "tclass"))
        class_dropdown.select_by_value(str(class_num))
        time.sleep(3)
        subject_dropdown = Select(driver.find_element(By.NAME, "tsubject"))
        subject_dropdown.select_by_visible_text("Science")
        time.sleep(3)
        book_dropdown = Select(driver.find_element(By.NAME, "tbook"))
        book_dropdown.select_by_visible_text("Science")
        time.sleep(3)

        # Click "Go"
        go_button = driver.find_element(By.XPATH, "//input[@value='Go']")
        go_button.click()
        time.sleep(5)

        # Wait for the sidebar containing download links
        wait = WebDriverWait(driver, 30)
        try:
            sidebar_container = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//td[@class='sidebar-menu']")
                )
            )
        except Exception as e:
            print(f"Sidebar container not found for Class {class_num}: {e}")
            return

        # Get the download link (last link in the sidebar)
        links = sidebar_container.find_elements(By.TAG_NAME, "a")
        if links:
            download_link = links[-1]
            download_url = download_link.get_attribute("href")
            print(f"Found download link for Class {class_num}: {download_url}")
            download_link.click()
            # Wait for the file to download (at least one minute)
            time.sleep(60)
            zip_path = self.rename_latest_zip(class_num)
            if zip_path:
                self.unzip_for_grade(class_num, zip_path)
        else:
            print(f"No download link found for Class {class_num}.")

        # Navigate back to the original URL for the next class
        driver.get(self.url)
        time.sleep(3)

    def download_for_classes(self, start: int, end: int):
        """
        Loop over and process downloads for classes from start to end (inclusive).
        """
        for class_num in range(start, end + 1):
            self.download_for_class(class_num)
            print("-" * 40)

    def quit(self):
        """
        Quit the WebDriver.
        """
        self.driver.quit()


def main():
    geckodriver_path = "/snap/bin/firefox.geckodriver"  # Update this path as needed
    download_dir = "/home/shtlp_0133/Documents/RAG Assignment 17 March/Final Folder/PDFs"  # Download folder
    extract_dir = "/home/shtlp_0133/Documents/RAG Assignment 17 March/Final Folder/Extracted"  # Folder for extracted files
    downloader = NCERTDownloader(geckodriver_path, download_dir, extract_dir)
    try:
        # Process classes 7 through 10
        downloader.download_for_classes(7, 10)
    finally:
        downloader.quit()


if __name__ == "__main__":
    main()
