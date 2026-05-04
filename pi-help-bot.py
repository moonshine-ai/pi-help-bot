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
import subprocess
import sys
import time
import Levenshtein
import netifaces as ni
import string
from pathlib import Path
from moonshine_voice import (
    EmbeddingModelArch,
    ModelArch,
    SPELLED,
    Dialog,
    DialogFlow,
    IntentRecognizer,
    MicTranscriber,
    TextToSpeech,
    spell_out,
)

DATA_DIR = Path(__file__).parent / "data"


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


def _ssh_active() -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "ssh"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ssh_port() -> int:
    try:
        with open("/etc/ssh/sshd_config", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Port "):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 22


def _current_wifi_ssid() -> str:
    try:
        result = subprocess.run(
            ["iwgetid", "-r"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _find_local_ip() -> str | None:
    for iface in ni.interfaces():
        addrs = ni.ifaddresses(iface)
        if ni.AF_INET not in addrs:
            continue
        for addr_info in addrs[ni.AF_INET]:
            ip = addr_info.get("addr", "")
            if ip and not ip.startswith("127."):
                return ip
    return None


def _scan_wifi_networks() -> list[str]:
    update_result = subprocess.run(
        ["sudo", "nmcli", "device", "wifi", "rescan"],
        capture_output=True, text=True, timeout=2,
    )
    print(f"[DEBUG] nmcli stdout: {update_result.stdout}, {update_result.stderr}", file=sys.stderr, flush=True)
    if update_result.returncode != 0:
        print(f"[ERROR] nmcli stderr: {update_result.stderr}", file=sys.stderr)
        return []
    scan_result = subprocess.run(
        ["nmcli", "-f", "SSID,SIGNAL", "device", "wifi", "list"],
        capture_output=True, text=True, timeout=2,
    )
    if scan_result.returncode != 0:
        print(f"[ERROR] nmcli stderr: {scan_result.stderr}", file=sys.stderr)
        return []
    networks = []
    print(f"[DEBUG] nmcli stdout: {scan_result.stdout}", file=sys.stderr, flush=True)
    for line in scan_result.stdout.strip().split("\n")[1:]:
        match = re.match(r"^(.*?)\s+(\d+)$", line.rstrip())
        if match:
            ssid_candidate = match.group(1).strip()
            signal = int(match.group(2).strip())
            if ssid_candidate:  # filter out blank SSIDs
                networks.append((ssid_candidate, signal))
    # Sort by descending signal strength
    networks.sort(key=lambda tup: tup[1], reverse=True)
    return [ssid for ssid, _ in networks[:20]]

def fuzzy_match_network(ssid: str, networks: list[str]) -> str | None:
    for desperation in range(3):
        for network in networks:
            if fuzzy_match(ssid, network, desperation):
                return network
    return None

def normalize(input: str) -> str:
    return input.lower()

def fuzzy_match(ssid: str, network: str, desperation: int = 0) -> bool:
    normalized_ssid = normalize(ssid)
    normalized_network = normalize(network)
    if normalized_ssid == normalized_network:
        return True
    if desperation > 0:
        levenshtein_distance = Levenshtein.distance(normalized_ssid, normalized_network)
        max_len = max(len(normalized_ssid), len(normalized_network))
        if max_len > 0 and levenshtein_distance / max_len <= 0.34:
            return True
    if desperation > 1:
        if normalized_network.startswith(normalized_ssid):
            return True
    return False
    
def add_config_commands(dialog_flow: DialogFlow, tts: TextToSpeech) -> None:

    def report_ip_address(d: Dialog):
        ip = _find_local_ip()
        if ip is None:
            yield d.say("Sorry, I couldn't find a local IP address.")
            return
        speech_ip = re.sub(r"(\d)", r"\1 ", ip.replace(".", " dot "))
        print(f"[DEBUG] reporting IP {ip!r} as {speech_ip!r}", file=sys.stderr)
        yield d.say(
            f"Okay. Your local IP address is {speech_ip}. "
            f"To repeat, that's {speech_ip}."
        )

    dialog_flow.register_flow("What is my IP address?", report_ip_address)

    def report_ssh_status(d: Dialog):
        if not _ssh_active():
            yield d.say(
                "S S H is not enabled on this Raspberry Pi. "
                "To enable it, say turn on S S H."
            )
            return
        port = _ssh_port()
        if port == 22:
            yield d.say("S S H is enabled and running on the default port 22.")
        else:
            yield d.say(f"S S H is enabled and running on port {port}.")

    dialog_flow.register_flow("Is ssh enabled?", report_ssh_status)

    def enable_ssh(d: Dialog):
        if _ssh_active():
            yield d.say("S S H is already enabled.")
            return
        if not (yield d.confirm("This will enable S S H on this Raspberry Pi. Are you sure?")):
            yield d.say("Okay, leaving S S H disabled.")
            return
        yield d.say("Enabling S S H now.")
        try:
            result = subprocess.run(
                ["sudo", "raspi-config", "nonint", "do_ssh", "0"],
                capture_output=True, text=True, timeout=15,
            )
        except FileNotFoundError:
            yield d.say("Sorry, raspi-config was not found on this system.")
            return
        except subprocess.TimeoutExpired:
            yield d.say("Sorry, the command timed out while trying to enable S S H.")
            return
        if result.returncode == 0:
            yield d.say("S S H has been enabled successfully.")
        else:
            print(f"[ERROR] raspi-config stderr: {result.stderr}", file=sys.stderr)
            yield d.say("Sorry, I wasn't able to enable S S H.")

    dialog_flow.register_flow("Turn on SSH", enable_ssh)

    def report_wifi_status(d: Dialog):
        ssid = _current_wifi_ssid()
        if ssid:
            yield d.say(f"Wi-Fi is connected to {ssid}.")
        else:
            yield d.say("Wi-Fi is not connected on this Raspberry Pi.")

    dialog_flow.register_flow("Is Wi-Fi connected?", report_wifi_status)

    def connect_to_wifi(d: Dialog):
        networks = _scan_wifi_networks()
        ssid = yield d.ask("What's the name of your Wi-Fi network? Say list if you want to pick from a list or spell if you want to spell out the start of the name")
        ssid = ssid.strip()

        if ssid.lower().strip(string.punctuation) == "list":
            yield d.say("Say yes to the network you want to connect to.")
            for network in networks:
                if (yield d.confirm(f"{network}?")):
                    ssid = network
                    break
        elif ssid.lower().strip(string.punctuation) == "spell":
            spell = yield d.ask("Spell out the start of the network name.", mode=SPELLED)
            for network in networks:
                if fuzzy_match(spell, network):
                    ssid = network
                    break

        matching_ssid = fuzzy_match_network(ssid, networks)
        if matching_ssid is None:
            yield d.say(f"Sorry, I couldn't find a matching network for {ssid}.")
            return

        password = yield d.ask(
            f"Please spell the Wi-Fi password for {matching_ssid} one character at a time, and say done when finished.",
            mode=SPELLED,
        )

        if (yield d.confirm("Would you like me to read the password back?")):
            yield d.say("I heard: " + " ".join(spell_out(password)))

        if not (yield d.confirm(f"Connect to {ssid} now?")):
            yield d.say("Okay, nothing changed.")
            return

        yield d.say(f"Connecting to {ssid}.")
        try:
            result = subprocess.run(
                ["sudo", "nmcli", "device", "wifi", "connect", ssid, "password", password],
                capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            yield d.say("Sorry, network manager was not found on this system.")
            return
        except subprocess.TimeoutExpired:
            yield d.say("Sorry, the connection attempt timed out.")
            return
        if result.returncode == 0:
            yield d.say(f"Connected to {ssid}.")
        else:
            print(f"[ERROR] nmcli stderr: {result.stderr}", file=sys.stderr)
            yield d.say(
                "Sorry, I wasn't able to connect. "
                "Please check the network name and password and try again."
            )

    dialog_flow.register_flow("Connect to Wi-Fi", connect_to_wifi)

    dialog_flow.register_global("cancel", lambda d: d.cancel())
    dialog_flow.register_global("start over", lambda d: d.restart())


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
        "--tts-voice",
        type=str,
        default="piper_en_US-amy-medium",
        help="Voice for text-to-speech (default: piper_en_US-amy-low)",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra moonshine_option_t entries; repeat for multiple (e.g. --option speed=1.1)",
    )
    parser.add_argument(
        "--log-io",
        action="store_true",
        help=(
            "Log every utterance received from the STT and every prompt "
            "spoken to the TTS to stderr in 'user: ...' / 'assistant: ...' "
            "format."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to the data directory (default: data)",
    )
    args = parser.parse_args()

    extra: dict[str, str | int | float | bool] = {}
    extra.update(parse_options_cli(args.option))

    if not args.data_dir.exists():
        print(f"[ERROR] Data directory '{args.data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loading embedding model (variant={args.variant})...", file=sys.stderr
    )
    print(f"args.data_dir: {args.data_dir}")
    embedding_model_path = args.data_dir / "download.moonshine.ai/model/embeddinggemma-300m"
    embedding_model_arch = EmbeddingModelArch.GEMMA_300M
    print(
        f"Creating intent recognizer from '{embedding_model_path}' (threshold={args.threshold})...",
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
    tts = TextToSpeech(args.tts_language, voice=args.tts_voice, asset_root=args.data_dir, options=extra)

    dialog_flow = DialogFlow(
        tts=tts,
        intent_recognizer=intent_recognizer,
        log_io=args.log_io,
    )
    add_config_commands(dialog_flow, tts)

    print(
        f"Loading transcription model (language={args.language})...",
        file=sys.stderr,
    )
    model_path = args.data_dir / "download.moonshine.ai/model/medium-streaming-en/quantized"
    model_arch = ModelArch.MEDIUM_STREAMING

    mic_transcriber = MicTranscriber(
        model_path=model_path, model_arch=model_arch
    )
    mic_transcriber.add_listener(dialog_flow)

    dialog_flow.say("Hello!")
    dialog_flow.say("I'm the Raspberry Pi Help Bot.")
    dialog_flow.say("How can I help you today?")

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
