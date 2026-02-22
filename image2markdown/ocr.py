from pathlib import Path
import ollama

DEFAULT_MODEL = "llama3.2-vision"

DEFAULT_PROMPT = """
You are an expert OCR system.
Extract all the text from this image and format it into clean, structured Markdown.
- Use appropriate Markdown headers (#, ##) based on the visual hierarchy of the text.
- Format lists correctly using bullet points or numbers.
- If there is tabular data, format it as a standard Markdown table.
- Output ONLY the Markdown text. Do not include any conversational filler.
""".strip()


def image_to_markdown(
    image_path: Path,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    image_path = image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
    )
    return response["message"]["content"]


def process_single_image(
    image_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
) -> None:
    md = image_to_markdown(image_path, model=model, prompt=prompt)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
