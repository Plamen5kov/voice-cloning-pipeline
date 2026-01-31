"""
count_words_sentences.py
Counts the total number of words and sentences in an EPUB file.
"""

import sys
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

def get_spine_ids(book):
    """Return the list of item ids in the spine (reading order)."""
    return [item[0] for item in book.spine]

def get_numbered_toc_titles(book):
    """Return TOC titles that start with a number (e.g., '14. Worlds Apart')."""
    def flatten_toc(toc):
        result = []
        for entry in toc:
            if isinstance(entry, epub.Link):
                result.append(entry.title.strip())
            elif isinstance(entry, tuple) and len(entry) > 0:
                result.extend(flatten_toc(entry))
        return result
    all_titles = flatten_toc(book.toc)
    numbered_titles = [t for t in all_titles if re.match(r'^\d+\.', t)]
    return numbered_titles

def extract_chapters(book, spine_ids, toc_titles, filter_func):
    """Yield (id, title, text) for each chapter in the spine with a TOC-matching title and non-empty content, using the filter_func."""
    allowed_titles = filter_func(toc_titles)
    # Build a mapping from normalized TOC main title to (full TOC title, chapter number)
    toc_title_map = {}
    for toc_title in allowed_titles:
        match = re.match(r'^(\d+)\.\s*(.*)', toc_title)
        if match:
            chapter_num, main_title = match.group(1), match.group(2).strip().lower()
            toc_title_map[main_title] = (toc_title, chapter_num)
        else:
            toc_title_map[toc_title.strip().lower()] = (toc_title, '?')
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT and item.get_id() in spine_ids:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text().strip()
            if not text:
                continue
            # Try to identify chapter title
            title = None
            for tag in ['h1', 'h2', 'title']:
                found = soup.find(tag)
                if found and found.get_text(strip=True):
                    title = found.get_text(strip=True)
                    break
            # Fuzzy match: check if any allowed TOC title is in the HTML title
            matched = False
            matched_chapter_num = '?'
            if title:
                for main_title, (full_toc_title, chapter_num) in toc_title_map.items():
                    if main_title in title.lower():
                        matched = True
                        matched_chapter_num = chapter_num
                        break
            if matched:
                yield (item.get_id(), title, text, matched_chapter_num)

def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def count_sentences(text):
    sentences = re.split(r'[.!?](?:\s|$)', text)
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

def get_full_toc(book):
    """Return a flat list of all TOC (Table of Contents) entries as (title, href) tuples."""
    def flatten_toc(toc):
        result = []
        for entry in toc:
            if isinstance(entry, epub.Link):
                result.append((entry.title.strip(), entry.href))
            elif isinstance(entry, tuple) and len(entry) > 0:
                result.extend(flatten_toc(entry))
        return result
    return flatten_toc(book.toc)

def numbered_chapter_filter(toc_titles):
    """Return a set of normalized titles for numbered chapters."""
    return set(toc_titles)

def even_chapter_filter(toc_titles):
    """Return a set of normalized titles for even-numbered chapters only."""
    even_titles = set()
    for t in toc_titles:
        match = re.match(r'^(\d+)\.', t)
        if match and int(match.group(1)) % 2 == 0:
            even_titles.add(t)
    return even_titles

def main(epub_path):
    book = epub.read_epub(epub_path)
    spine_ids = get_spine_ids(book)
    toc_titles = get_numbered_toc_titles(book)
    full_toc = get_full_toc(book)
    print("Full Table of Contents (TOC):")
    for t, _ in full_toc:
        print(f"- {t}")
    print("\nNumbered TOC titles:")
    for t in toc_titles:
        print(f"- {t}")
    print("\nChapters included in count (numbered chapters):")
    all_text = []
    for cid, title, text, chapter_num in extract_chapters(book, spine_ids, toc_titles, numbered_chapter_filter):
        print(f"- {chapter_num}: {title}")
        all_text.append(text)
    joined_text = '\n'.join(all_text)
    print(f"\nTotal words: {count_words(joined_text)}")
    print(f"Total sentences: {count_sentences(joined_text)}")

    print("\nChapters included in count (even chapters):")
    all_text_even = []
    for cid, title, text, chapter_num in extract_chapters(book, spine_ids, toc_titles, even_chapter_filter):
        print(f"- {chapter_num}: {title}")
        all_text_even.append(text)
    joined_text_even = '\n'.join(all_text_even)
    print(f"\nTotal words (even chapters): {count_words(joined_text_even)}")
    print(f"Total sentences (even chapters): {count_sentences(joined_text_even)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_words_sentences.py <path_to_epub_file>")
        sys.exit(1)
    main(sys.argv[1])
