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
import sounddevice as sd
import string
from pathlib import Path
import pyudev
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
    print(
        f"[DEBUG] nmcli stdout: {update_result.stdout}, {update_result.stderr}", file=sys.stderr, flush=True)
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
    print(f"[DEBUG] nmcli stdout: {scan_result.stdout}",
          file=sys.stderr, flush=True)
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
    print(
        f"[DEBUG] fuzzy_match_network: ssid: {ssid}, networks: {networks}", file=sys.stderr)
    if ssid is None:
        return None
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
        levenshtein_distance = Levenshtein.distance(
            normalized_ssid, normalized_network)
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

    def set_ssh(d: Dialog, enable: bool):
        verb = "enable" if enable else "disable"
        gerund = "Enabling" if enable else "Disabling"
        past = "enabled" if enable else "disabled"
        opposite_past = "disabled" if enable else "enabled"
        if _ssh_active() == enable:
            yield d.say(f"S S H is already {past}.")
            return
        if not (yield d.confirm(f"This will {verb} S S H on this Raspberry Pi. Are you sure?")):
            yield d.say(f"Okay, leaving S S H {opposite_past}.")
            return
        yield d.say(f"{gerund} S S H now.")
        try:
            result = subprocess.run(
                ["sudo", "raspi-config", "nonint",
                    "do_ssh", "0" if enable else "1"],
                capture_output=True, text=True, timeout=15,
            )
        except FileNotFoundError:
            yield d.say("Sorry, raspi-config was not found on this system.")
            return
        except subprocess.TimeoutExpired:
            yield d.say(f"Sorry, the command timed out while trying to {verb} S S H.")
            return
        if result.returncode == 0:
            yield d.say(f"S S H has been {past} successfully.")
        else:
            print(
                f"[ERROR] raspi-config stderr: {result.stderr}", file=sys.stderr)
            yield d.say(f"Sorry, I wasn't able to {verb} S S H.")

    dialog_flow.register_flow("Turn on SSH", lambda d: set_ssh(d, True))
    dialog_flow.register_flow("Turn off SSH", lambda d: set_ssh(d, False))

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
            ssid = yield d.ask("Spell out the start of the network name.", mode=SPELLED)
            print(f"[DEBUG] spelled buffer: {ssid!r}", file=sys.stderr)

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

        if not (yield d.confirm(f"Connect to {matching_ssid} now?")):
            yield d.say("Okay, nothing changed.")
            return

        yield d.say(f"Connecting to {matching_ssid}.")
        try:
            result = subprocess.run(
                ["sudo", "nmcli", "device", "wifi",
                    "connect", ssid, "password", password],
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


_VIRTUAL_AUDIO_PREFIXES = (
    'default', 'sysdefault', 'front', 'rear', 'center_lfe', 'side',
    'surround', 'iec958', 'spdif', 'hdmi', 'modem', 'phoneline',
    'dmix', 'dsnoop', 'pulse', 'pipewire', 'jack', 'oss',
    'null', 'samplerate', 'speex', 'upmix', 'vdownmix',
)
# Real hardware always carries an (hw:CARD,DEV) or (plughw:...) tag.
_HW_AUDIO_RE = re.compile(r'\((?:hw|plughw):(\d+),\d+\)')


def _is_real_audio_hardware(d) -> bool:
    if d['name'].startswith(_VIRTUAL_AUDIO_PREFIXES):
        return False
    return bool(_HW_AUDIO_RE.search(d['name']))


def is_real_audio_input(d):
    if d['max_input_channels'] <= 0:
        return False
    return _is_real_audio_hardware(d)


def is_real_audio_output_only(d):
    if d['max_output_channels'] <= 0:
        return False
    if d['max_input_channels'] > 0:
        return False
    return _is_real_audio_hardware(d)


def is_real_audio_output(d):
    """Real audio hardware with output capability (output-only OR input+output)."""
    if d['max_output_channels'] <= 0:
        return False
    return _is_real_audio_hardware(d)


def find_best_tts_output_device() -> int | None:
    """Return the index of the best TTS output device, or None.

    Prefers real output-only hardware over real input+output hardware so a
    hot-plugged speaker still wins over a headset, but falls back to a
    headset (or any other real output-capable device) when no output-only
    device is connected.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[DEBUG] sounddevice query failed: {e}", file=sys.stderr)
        return None
    output_only_idx: int | None = None
    input_output_idx: int | None = None
    for idx, device in enumerate(devices):
        if not is_real_audio_output(device):
            continue
        if device['max_input_channels'] == 0:
            if output_only_idx is None:
                output_only_idx = idx
        else:
            if input_output_idx is None:
                input_output_idx = idx
    chosen = output_only_idx if output_only_idx is not None else input_output_idx
    if chosen is not None:
        kind = "output-only" if chosen == output_only_idx else "input+output"
        print(
            f"[DEBUG] Best TTS output device at index {chosen} "
            f"({kind}): {devices[chosen]['name']!r}",
            file=sys.stderr,
        )
    return chosen


def _output_capable_device_names() -> set[str]:
    try:
        return {d['name'] for d in sd.query_devices() if is_real_audio_output(d)}
    except Exception as e:
        print(f"[DEBUG] sounddevice query failed: {e}", file=sys.stderr)
        return set()


def wait_for_input_device(retry_count: int = None) -> int:
    """Poll sounddevice until an audio input device is available.

    Re-initializes PortAudio on each poll so that newly hot-plugged devices
    are picked up. Sleeps one second between polls. Returns the sounddevice
    device index of the first device with input channels.
    """
    while retry_count is None or retry_count > 0:
        try:
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if is_real_audio_input(device):
                    print(
                        f"[DEBUG] Found audio input device at index {idx}: "
                        f"{device['name']!r}",
                        file=sys.stderr,
                    )
                    return idx
        except Exception as e:
            print(f"[DEBUG] sounddevice query failed: {e}", file=sys.stderr)
        print(
            "[DEBUG] No audio input device available; sleeping 1s before next poll...",
            file=sys.stderr,
        )
        time.sleep(1)
        if retry_count is not None:
            retry_count -= 1
    print(
        f"[ERROR] No audio input device available after {retry_count} retries.", file=sys.stderr)
    return None


def refresh_devices():
    # Force PortAudio to re-enumerate; otherwise sd.query_devices()
    # keeps returning the snapshot from process start.
    sd._terminate()
    sd._initialize()


# A single USB sound card registers separate ALSA PCM nodes for capture and
# playback (pcmC*D*c / pcmC*D*p), so each physical plug fires two udev events
# with 'pcm' in the device_node. Collapse them into one logical event.
_HOTPLUG_DEBOUNCE_SECONDS = 0.5
_last_hotplug_event_time = {'add': 0.0, 'remove': 0.0}


def on_event(action, device):
    # 'sound' fires for cards, controls, pcm nodes — you'll get several
    # events per USB plug-in. Debounce per action so we only react once.
    global audio_input_changed, audio_output_changed, known_output_capable_devices
    if action not in ('add', 'remove'):
        return
    if not device.device_node or 'pcm' not in (device.device_node or ''):
        return

    now = time.monotonic()
    if now - _last_hotplug_event_time[action] < _HOTPLUG_DEBOUNCE_SECONDS:
        return
    _last_hotplug_event_time[action] = now

    refresh_devices()
    devices = sd.query_devices()
    inputs = [d for d in devices if d['max_input_channels'] > 0]
    if action == 'add':
        print('New input devices:', inputs)
    else:
        print('Removed input devices:', inputs)

    current_output_capable = {d['name'] for d in devices if is_real_audio_output(d)}
    added = current_output_capable - known_output_capable_devices
    removed = known_output_capable_devices - current_output_capable
    if added:
        print(f'Newly hot-plugged output-capable devices: {sorted(added)}')
    if removed:
        print(f'Unplugged output-capable devices: {sorted(removed)}')
    if added or removed:
        # Re-pick TTS output: a fresh device may be a better candidate,
        # and a removed device may have been the one we were using.
        audio_output_changed = True
    known_output_capable_devices = current_output_capable

    audio_input_changed = True


def setup_hotplug_event_handler():
    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem='sound')
    observer = pyudev.MonitorObserver(monitor, on_event)
    observer.start()


def main() -> None:
    global audio_input_changed, audio_output_changed, known_output_capable_devices
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
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        help="Audio output device to use (default: None)",
    )
    parser.add_argument(
        "--output-volume",
        type=float,
        default=1.0,
        help="Audio output volume to use (default: 1.0)",
    )

    args = parser.parse_args()

    extra: dict[str, str | int | float | bool] = {}
    extra.update(parse_options_cli(args.option))

    if not args.data_dir.exists():
        print(
            f"[ERROR] Data directory '{args.data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loading embedding model (variant={args.variant})...", file=sys.stderr
    )
    print(f"args.data_dir: {args.data_dir}")
    embedding_model_path = args.data_dir / \
        "download.moonshine.ai/model/embeddinggemma-300m"
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
    tts = TextToSpeech(args.tts_language, voice=args.tts_voice, asset_root=args.data_dir,
                       output_device=args.output_device, volume=args.output_volume, options=extra)

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
    model_path = args.data_dir / \
        "download.moonshine.ai/model/medium-streaming-en/quantized"
    model_arch = ModelArch.MEDIUM_STREAMING

    audio_input_changed = False
    audio_output_changed = False
    # Seed the snapshot so we only react to output-capable devices that
    # appear AFTER startup (i.e. truly hot-plugged ones).
    refresh_devices()
    known_output_capable_devices = _output_capable_device_names()
    setup_hotplug_event_handler()
    audio_device = wait_for_input_device(retry_count=1)
    if audio_device is None:
        print(f"No audio input device available; waiting for one...", file=sys.stderr)
        while True:
            audio_device = wait_for_input_device(retry_count=1)
            if audio_device is not None:
                break
            time.sleep(1)
            print(
                f"No audio input device available; sleeping 1s before next poll...", file=sys.stderr)
        print(
            f"Audio input device found after waiting: {audio_device}", file=sys.stderr)
    while True:
        mic_transcriber = MicTranscriber(
            model_path=model_path, model_arch=model_arch, device=audio_device
        )
        mic_transcriber.add_listener(dialog_flow)

        # Apply any pending output-device switch before the greetings are
        # queued — TextToSpeech.say() snapshots ``_output_device`` at
        # enqueue time, so updating it after the say() calls would not
        # affect the already-queued utterances.
        if audio_output_changed:
            new_output_device = find_best_tts_output_device()
            if new_output_device is not None and new_output_device != tts._output_device:
                print(
                    f"Switching TTS output to device {new_output_device}",
                    file=sys.stderr,
                )
                tts._output_device = new_output_device
                tts._say_device_cache = None
                tts._say_settings_ok = None
            audio_output_changed = False

        dialog_flow.say("Hello!")
        dialog_flow.say("I'm the Raspberry Pi Help Bot.")
        dialog_flow.say("How can I help you today?")

        print("\n" + "=" * 60, file=sys.stderr)
        print("Raspberry Pi Help Bot", file=sys.stderr)
        print(f"Audio device: {audio_device}", file=sys.stderr)
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
                if audio_output_changed:
                    new_output_device = find_best_tts_output_device()
                    if new_output_device is not None and new_output_device != tts._output_device:
                        print(
                            f"Switching TTS output to device {new_output_device}",
                            file=sys.stderr,
                        )
                        tts._output_device = new_output_device
                        # Force the TTS playback worker to re-resolve the
                        # device + sample-rate on its next utterance.
                        tts._say_device_cache = None
                        tts._say_settings_ok = None
                    audio_output_changed = False
                if audio_input_changed:
                    print(
                        f"Audio input changed: {audio_input_changed}", file=sys.stderr)
                    # Debounce multiple events in a row.
                    time.sleep(1)
                    # Wait for a new audio input device to be available.
                    audio_device = wait_for_input_device()
                    audio_input_changed = False
                    break
        except KeyboardInterrupt:
            print("\nStopping...", file=sys.stderr)
            return
        finally:
            mic_transcriber.stop()
            mic_transcriber.close()


if __name__ == "__main__":
    main()
