import re

def fixed_size_chunk(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start = start + chunk_size - overlap
    return chunks

def split_by_structure(text):
    pattern = r"(SECTION|SEC\.|ARTICLE|Article|Chapter|Clause)\s+\d+"
    matches = list(re.finditer(pattern, text))

    if len(matches) == 0:
        return [text]

    sections = []

    # capture any text before the first marker (preamble/title)
    first_marker_start = matches[0].start()
    if first_marker_start > 0:
        preamble = text[:first_marker_start]
        if preamble.strip():            # only if it's not just whitespace
            sections.append(preamble)

    for i in range(len(matches)):
        start = matches[i].start()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)
        sections.append(text[start:end])
    return sections

def hybrid_chunk(text, max_section_size=1500):
    sections = split_by_structure(text)
    final_chunks = []
    for section in sections:
        if len(section) <= max_section_size:
            final_chunks.append(section)
        else:
            smaller = fixed_size_chunk(section)
            final_chunks.extend(smaller)
    return final_chunks

if __name__ == "__main__":
    test = "SECTION 1. Short part. SECTION 2. " + ("X" * 3000)
    result = hybrid_chunk(test)
    print("Number of final chunks:", len(result))
    for c in result:
        print("Chunk length:", len(c))