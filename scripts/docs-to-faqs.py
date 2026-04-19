"""
docs-to-faqs.py — Generate FAQ training data from Raspberry Pi AsciiDoc sources.

Walks a directory of .adoc files, expands include:: directives, and sends
each file's content to an LLM (Ollama or Anthropic) to produce spoken-style
Q:/A: pairs suitable for training a voice assistant.

Setup:
    pip install ollama          # for local Ollama
    pip install anthropic       # for Claude (needs ANTHROPIC_API_KEY)

Run (all .adoc files under the default documentation/documentation tree):
    python docs-to-faqs.py --anthropic

Run (a specific subdirectory):
    python docs-to-faqs.py --adoc documentation/documentation/asciidoc/computers/os --anthropic

Output:
    A sibling <stem>.faq.txt next to each .adoc file (Q:/A: lines).
    Use --force to overwrite existing .faq.txt files.
"""

import argparse
import re
import time
from pathlib import Path

try:
    from ollama import chat as ollama_chat
except ImportError:
    ollama_chat = None

try:
    import anthropic
except ImportError:
    anthropic = None

DEFAULT_OLLAMA_FAQ_MODEL = "gemma3:27b"

# include directive: include::path/to/file.adoc[...]
_INCLUDE_RE = re.compile(r"^include::([^\[]+)\[")


def _resolve_includes(adoc_path: Path, depth: int = 0) -> list[str]:
    """
    Recursively expand include:: directives, returning a flat list of lines.
    Limits recursion to 8 levels to guard against cycles.
    """
    if depth > 8 or not adoc_path.exists():
        return []

    lines = adoc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = []
    for line in lines:
        m = _INCLUDE_RE.match(line)
        if m:
            include_path = adoc_path.parent / m.group(1)
            result.extend(_resolve_includes(include_path, depth + 1))
        else:
            result.append(line)
    return result


FAQ_SYSTEM_PROMPT = """
You are generating training questions for a voice assistant that answers Raspberry Pi questions.

You will be given a section of Raspberry Pi documentation. Generate questions that real users would actually SAY OUT LOUD to a voice assistant — the kind of casual, sometimes frustrated questions a beginner would speak, not type.

STYLE RULES:
- Write like someone talking, not typing. Use contractions (don't, I've, it's, can't).
- Many good questions start with a situation or problem, then ask the question:
  "My audio is coming out of HDMI but I want it through the headphone jack. How do I change that?"
  "I installed the Lite version and there's no media player. What do I need to get one?"
- Avoid formal question openers: never start with "Is it possible to", "What option", "Can I tell the player to", "What command should I use to", "How does one".
- Use plain words, not doc words. Say "flags" not "CLI options", say "headphone jack" not "audio output device", say "close when it's done" not "terminate upon completion".
- It's fine for a question to be incomplete or slightly rambly, like someone thinking out loud:
  "I just want the video to go fullscreen automatically. Is there a flag for that or something?"
  "Wait, can I play audio and video at the same time to different outputs?"
- Mix question lengths: some short ("Does VLC work on Pi Lite?"), some longer with context.
- Include beginner-level confusion questions, not just how-to questions:
  "I'm not sure if I have VLC installed. How do I check?"
  "What even is ALSA? Do I need to know about it to change my audio output?"

BAD examples (too formal, don't write like this):
  "What command should I run to see a list of all available audio devices?"
  "Can I navigate to a file directly from the Media menu option?"
  "Is there a way to run this without opening any graphical windows at all?"

GOOD examples (casual spoken style):
  "How do I see what audio devices I've got?"
  "I just want to open a video without using the terminal. Can I do that?"
  "Can I run it without a desktop, like just from the command line?"

OUTPUT FORMAT:
- Group items under the relevant section heading as a markdown heading line starting with #, plus a "# Whole Document" section for questions that apply to the whole passage.
- For each hypothetical question, output exactly two lines:
  Q: <the question — same casual spoken style as above>
  A: <one line; exactly 10 to 15 words (count carefully). Summarize only information that appears in the source text for that section (or in the full passage for Whole Document items). Do not add facts, guesses, or general knowledge not stated there. If the text does not support a concise answer, write a short refusal like "Not specified in this section.">
- Leave one blank line between Q/A pairs if you like, but keep Q: then A: adjacent for each item.
- Do not refer to "the document" or "the text" in the questions themselves."""

def _faq_system_prompt() -> str:
    return FAQ_SYSTEM_PROMPT.strip()


_OLLAMA_USER_WRAPPER = """=== TASK (obey exactly; this is not a chat) ===

__INSTRUCTIONS_PLACEHOLDER__

=== SOURCE MATERIAL (generate Q/A lines from this only) ===

__SOURCE_PLACEHOLDER__
"""


def generate_faq_ollama(
    adoc_content: str,
    model: str = DEFAULT_OLLAMA_FAQ_MODEL,
) -> str:
    """
    Ask an Ollama model for Q:/A: pairs from AsciiDoc source content.

    Puts full instructions in the user message because many Ollama models
    ignore or weakly weight the system role.
    """
    if ollama_chat is None:
        raise RuntimeError("ollama package is not installed. pip install ollama")

    instructions = _faq_system_prompt()
    instructions += """

STRICT OUTPUT RULES (Ollama):
- Your entire reply must be ONLY the FAQ text: # section headings, then Q:/A: lines. Nothing else.
- Do NOT write an introduction, overview, "structured summary", or "here is a summary".
- Do NOT offer to help, ask follow-up questions, or list numbered options.
- Do NOT describe the document or author before the questions; start with the first # heading or Q: line as required by the format above.
"""

    user_message = (
        _OLLAMA_USER_WRAPPER.replace("__INSTRUCTIONS_PLACEHOLDER__", instructions)
        .replace("__SOURCE_PLACEHOLDER__", adoc_content)
    )

    system_guard = (
        "You are a formatting engine. Follow the TASK section in the user message exactly. "
        "Output only the FAQ format (Q:/A: and # headings); no preamble or summary."
    )

    ollama_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_guard},
            {"role": "user", "content": user_message},
        ],
    )
    try:
        response = ollama_chat(**ollama_kwargs, options={"temperature": 0.35})
    except TypeError:
        response = ollama_chat(**ollama_kwargs)

    if hasattr(response, "message"):
        return response.message.content or ""
    return (response.get("message") or {}).get("content", "")


def generate_faq_anthropic(
    adoc_content: str,
    adoc_path: Path,
    model: str = "claude-sonnet-4-6",
    force: bool = False,
):
    """Generate Q:/A: pairs using Anthropic's Claude API and write to .faq.txt."""
    if anthropic is None:
        raise RuntimeError(
            "anthropic package is not installed. pip install anthropic"
        )

    system_prompt = _faq_system_prompt()

    output_file = adoc_path.with_suffix(".faq.txt")
    if output_file.exists() and not force:
        print(f"[faq] FAQ already exists for {adoc_path}")
        return

    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_prompt,
                cache_control={"type": "ephemeral"},
                messages=[{"role": "user", "content": adoc_content}],
            )
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            print(f"[faq] Error generating FAQ for {adoc_path} (attempt {attempt + 1}/{max_attempts}): {e}")
            time_to_wait = 20 * (attempt + 1)
            for i in range(time_to_wait):
                print(f"[faq] Waiting {i + 1} of {time_to_wait} seconds...", end="\r")
                time.sleep(1)

    with open(output_file, "w") as f:
        f.write(response.content[0].text)


def run_faq_for_adoc(
    adoc_path: Path,
    model: str = DEFAULT_OLLAMA_FAQ_MODEL,
    use_anthropic: bool = False,
    force: bool = False,
) -> None:
    """Load an .adoc file (with includes resolved), then generate FAQ via Ollama or Anthropic."""
    lines = _resolve_includes(Path(adoc_path))
    if not lines:
        raise ValueError(f"[faq] No lines found in {adoc_path}")
    adoc_content = "\n".join(lines)

    if use_anthropic:
        generate_faq_anthropic(
            adoc_content,
            adoc_path,
            model=model,
            force=force,
        )
        return

    output_file = adoc_path.with_suffix(".faq.txt")
    if output_file.exists() and not force:
        print(f"[faq] FAQ already exists for {adoc_path}")
        return

    result = generate_faq_ollama(adoc_content, model=model)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"[faq] Wrote {output_file}")


def run_faq_for_all_adocs(
    adoc_dir: Path,
    model: str = DEFAULT_OLLAMA_FAQ_MODEL,
    use_anthropic: bool = False,
    force: bool = False,
) -> None:
    """Generate FAQ files for every .adoc file under adoc_dir."""
    for adoc_path in sorted(adoc_dir.rglob("*.adoc")):
        print(f"[faq] Generating FAQ for {adoc_path}")
        try:
            run_faq_for_adoc(
                adoc_path,
                model=model,
                use_anthropic=use_anthropic,
                force=force,
            )
        except Exception as e:
            print(f"[faq] Error generating FAQ for {adoc_path}: {e}")
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FAQ training data from Raspberry Pi AsciiDoc sources."
    )
    parser.add_argument(
        "--adoc",
        type=str,
        default="documentation/documentation",
        help="Directory of .adoc files to process recursively (default: documentation/documentation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_FAQ_MODEL,
        help=f"Ollama model (default: {DEFAULT_OLLAMA_FAQ_MODEL}) or Anthropic model when --anthropic (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--anthropic",
        action="store_true",
        help="Use Anthropic API (Claude) instead of Ollama; requires ANTHROPIC_API_KEY",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .faq.txt files instead of skipping",
    )
    args = parser.parse_args()

    adoc_dir = Path(args.adoc)
    if not adoc_dir.is_dir():
        parser.error(f"Not a directory: {adoc_dir}")

    model = args.model
    if args.anthropic and model == DEFAULT_OLLAMA_FAQ_MODEL:
        model = "claude-sonnet-4-6"

    run_faq_for_all_adocs(
        adoc_dir,
        model=model,
        use_anthropic=args.anthropic,
        force=args.force,
    )


if __name__ == "__main__":
    main()
