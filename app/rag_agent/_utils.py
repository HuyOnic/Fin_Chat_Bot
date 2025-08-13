def load_prompt_sections(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = {}
    current_key = None
    for line in content.splitlines():
        if line.startswith("===") and line.endswith("==="):
            current_key = line.strip("=\n")
            sections[current_key] = []
        elif current_key:
            sections[current_key].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}
