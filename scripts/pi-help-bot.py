"""
help-bot.py — Voice-driven Raspberry Pi help bot.

Listens to the microphone, matches spoken questions against pre-computed
FAQ embeddings using Moonshine Voice's IntentRecognizer, and speaks the
answer aloud via TTS.

Setup:
    pip install moonshine-voice

Run:
    python scripts/help-bot.py
    python scripts/help-bot.py --embeddings path/to/embeddings.json
    python scripts/help-bot.py --threshold 0.6 --variant q4
"""

import argparse
import json
import re
import sys
import time
import netifaces as ni
from moonshine_voice import (
    IntentRecognizer,
    MicTranscriber,
    TextToSpeech,
    TranscriptEventListener,
    get_embedding_model,
    get_model_for_language,
)


def parse_options_cli(options: list[str]) -> dict[str, str | int | float | bool]:
    extra: dict[str, str | int | float | bool] = {}
    for option in options:
        k, v = option.split("=", 1)
        extra[k] = v
    return extra


def load_doc_faqs(embeddings_path: str, intent_recognizer: IntentRecognizer, tts: TextToSpeech) -> dict[str, str]:
    print(f"Loading embeddings from {embeddings_path}...", file=sys.stderr)
    with open(embeddings_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} FAQ entries.", file=sys.stderr)

    answers = {}
    for entry in entries:
        question = entry["question"]
        answer = entry.get("answer", "")
        embedding = entry.get("embedding")

        if not answer or not embedding:
            continue

        answers[question] = answer

        def make_handler(q, a):
            def handler(trigger: str, utterance: str, similarity: float):
                print(f"\n--- Intent Match ---", file=sys.stderr)
                print(f"  Utterance:  {utterance}", file=sys.stderr)
                print(f"  Matched Q:  {trigger}", file=sys.stderr)
                print(f"  Similarity: {similarity:.2%}", file=sys.stderr)
                print(f"  Answer:     {a}", file=sys.stderr)
                print(f"--------------------", file=sys.stderr)
                tts.say(a)

            return handler

        intent_recognizer.register_intent(
            question,
            make_handler(question, answer),
            embedding=embedding,
        )

    print(
        f"Registered {intent_recognizer.intent_count} intents.", file=sys.stderr
    )

    def on_any_intent(match):
        print(
            f"[DEBUG] on_intent: canonical={match.canonical_phrase!r} "
            f"utterance={match.utterance!r} similarity={match.similarity:.4f}",
            file=sys.stderr,
        )

    intent_recognizer.set_on_intent(on_any_intent)


def add_config_commands(intent_recognizer: IntentRecognizer, tts: TextToSpeech) -> None:

    def report_ip_address(trigger: str, utterance: str, similarity: float) -> None:
        for iface in ni.interfaces():
            addrs = ni.ifaddresses(iface)
            if ni.AF_INET not in addrs:
                continue
            for addr_info in addrs[ni.AF_INET]:
                ip = addr_info.get("addr", "")
                if ip and not ip.startswith("127."):
                    speech_ip = re.sub(
                        r"(\d)", r"\1 ", ip.replace(".", " dot "))
                    speech = [
                        "Okay",
                        f"Your local IP address is {speech_ip}",
                        f"To repeat, that's {speech_ip}.",
                    ]
                    break
        if speech is None:
            speech = [
                "Sorry",
                "I couldn't find a local IP address.",
            ]
        print(f"[DEBUG] {speech}", file=sys.stderr)
        tts.say(speech, speed=0.75)

    intent_recognizer.register_intent(
        "What is my IP address?",
        report_ip_address,
        embedding=None,
    )


class TranscriptLogger(TranscriptEventListener):
    """Prints transcript events to stderr for debugging."""

    def on_line_started(self, event):
        print(f"[TRANSCRIPT] Started: {event.line.text}", file=sys.stderr)

    def on_line_text_changed(self, event):
        print(f"[TRANSCRIPT] Updated: {event.line.text}", file=sys.stderr)

    def on_line_completed(self, event):
        print(f"[TRANSCRIPT] Completed: {event.line.text}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice-driven Raspberry Pi help bot using Moonshine Voice."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="embeddings.json",
        help="Path to the pre-computed embeddings JSON file (default: embeddings.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for intent matching (default: 0.7)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="q4",
        choices=["q4", "q8", "fp16", "fp32", "q4f16"],
        help="Embedding model variant (default: q4)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for speech recognition (default: en)",
    )
    parser.add_argument(
        "--tts-language",
        type=str,
        default="en-us",
        help="Language/voice for text-to-speech (default: en-us)",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra moonshine_option_t entries; repeat for multiple (e.g. --option speed=1.1)",
    )
    args = parser.parse_args()

    extra: dict[str, str | int | float | bool] = {}
    extra.update(parse_options_cli(args.option))

    print(
        f"Loading embedding model (variant={args.variant})...", file=sys.stderr
    )
    embedding_model_path, embedding_model_arch = get_embedding_model(
        "embeddinggemma-300m", args.variant
    )

    print(
        f"Creating intent recognizer (threshold={args.threshold})...",
        file=sys.stderr,
    )
    intent_recognizer = IntentRecognizer(
        model_path=embedding_model_path,
        model_arch=embedding_model_arch,
        model_variant=args.variant,
        threshold=args.threshold,
    )

    print(
        f"Initializing TTS (language={args.tts_language})...", file=sys.stderr)
    tts = TextToSpeech(args.tts_language, options=extra)

    # load_doc_faqs(args.embeddings, intent_recognizer, tts)

    add_config_commands(intent_recognizer, tts)

    print(
        f"Loading transcription model (language={args.language})...",
        file=sys.stderr,
    )
    model_path, model_arch = get_model_for_language(args.language)

    mic_transcriber = MicTranscriber(
        model_path=model_path, model_arch=model_arch
    )
    mic_transcriber.add_listener(TranscriptLogger())
    mic_transcriber.add_listener(intent_recognizer)

    tts.say(["Hello!", "I'm the Raspberry Pi Help Bot.", "How can I help you today?"])

    print("\n" + "=" * 60, file=sys.stderr)
    print("Raspberry Pi Help Bot", file=sys.stderr)
    print(
        "Ask a question about Raspberry Pi and I'll answer it!",
        file=sys.stderr,
    )
    print("Press Ctrl+C to stop.", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    mic_transcriber.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    finally:
        intent_recognizer.close()
        mic_transcriber.stop()
        mic_transcriber.close()


if __name__ == "__main__":
    main()
