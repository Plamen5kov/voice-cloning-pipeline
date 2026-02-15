"""
read_text_file.py
Reads an EPUB file and prints its text content to the console.
"""


import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def read_epub_file(filepath):
    book = epub.read_epub(filepath)
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            print(text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python read_text_file.py <path_to_epub_file>")
    else:
        read_epub_file(sys.argv[1])
