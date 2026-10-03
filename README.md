# Verbatim

Turn any audio or video file into text, with each speaker labelled. Queue up as many recordings as you like and it works through them one after another. Everything runs on your own computer: no audio, text or file name is sent anywhere, and there is no account, key or subscription.

Windows and macOS.

## What it does

- **Any format in.** MP3, M4A, WAV, FLAC, OGG, Opus, WMA, AAC, AMR, AIFF, and the sound track of MP4, MKV, MOV, WebM, AVI and the rest. A static FFmpeg is built in, so nothing needs installing.
- **A queue.** Add files, add a whole folder, or drop them on the window. Add more while it is running. One failed file does not stop the rest.
- **Who said what.** Each voice becomes Speaker 1, Speaker 2 and so on. Tell it how many people there are if you know, rename them afterwards, and give two speakers the same name to merge them.
- **Noise handling and invented-text protection.** See [Keeping invented text out](#keeping-invented-text-out).
- **Seven outputs.** Plain text, Word (.docx), Markdown, SRT and VTT subtitles, CSV, and JSON with every timestamp and confidence score. Tick the ones you want and each transcript is saved in all of them.
- **Saved to Downloads.** Transcripts go to `Downloads/Verbatim` unless you choose another folder (File > Default save folder…). An existing file is never overwritten: the new one becomes `name (2).txt`.

## Which file do I use?

| I want to… | Windows | macOS |
|---|---|---|
| Build a standalone app (recommended) | `build-exe.bat` | `build-app.command` |
| Run from source without building | `run.bat` | `run.command` |

Both build files are double-click and install everything themselves: Python if it is missing (Homebrew too, on a Mac), every dependency, and the speech models. The first build downloads about 500 MB of models; they are kept in `build-models/` so later builds do not download them again.

- **Windows** produces the folder `dist\Verbatim` with `Verbatim.exe` inside. It is a folder rather than a single exe on purpose: roughly 700 MB of models are baked in, and a one-file exe would unpack all of it to a temporary folder at every launch. Move the whole folder wherever you like.
- **macOS** produces `dist/Verbatim.app`. Drag it to Applications.

Each build ends by running the app's self-test, which transcribes a short two-speaker sample for real, feeds it 20 seconds of noise and checks nothing is "transcribed", round-trips every export format and builds the window. It says `ALL GOOD` or `PROBLEMS FOUND`; the report is `verbatim-selftest.txt` next to the app.

On a Mac, a new `.command` file may need this once in Terminal before it will double-click:

```
cd ~/Downloads/GitHub/verbatim
chmod +x build-app.command run.command
xattr -d com.apple.quarantine build-app.command run.command 2>/dev/null
```

To bake in a different speech model, set `VERBATIM_BAKE` before building (`whisper-small`, `whisper-turbo`, `parakeet-v3`, …, or `none` to bake only the small helper models and let the app download a speech model on first use).

## Speech models

| Model | Languages | Size | Speed | Use it for |
|---|---|---|---|---|
| **Parakeet v2** (default, baked in) | English | 630 MB | fast | English. The most accurate here and the least likely to invent words. |
| Parakeet v3 | 25 European languages | 640 MB | fast | French, German, Spanish, Italian, Dutch, Polish and so on. Not Welsh. |
| Whisper large-v3 turbo | 99, including Welsh | 1 GB | slow | The best quality for any other language. |
| Whisper small | 99, including Welsh | 360 MB | medium | Other languages on a modest laptop. |
| Whisper base / tiny | 99 | 155 / 100 MB | fast | Rough drafts only. |

Models other than the baked-in one are downloaded once from **Models > Download and manage models…** and kept in a `verbatim-models` folder beside the app. They come from the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project's GitHub releases; nothing is fetched from anywhere else and no token is needed.

Whisper models can also translate into English, and take a fixed language if you would rather not rely on detection.

## Keeping invented text out

Speech models, Whisper especially, were trained on subtitled video. Hand them silence, music or hiss and they answer with what subtitles say at such moments: "Thanks for watching", a credit line, a lone "you", or one phrase looped forty times. None of it was said. Verbatim has four defences, in this order:

1. **Only speech is transcribed.** A speech detector (Silero VAD) marks where someone is actually talking, and only those stretches ever reach the speech model. Silence, music and background noise are never shown to it.
2. **Noise reduction when it is needed.** The background level is measured from the gaps between speech. If the recording is noisy, the file goes through a noise-reduction model (GTCRN) and speech is detected again on the cleaned audio, so noise bursts are not mistaken for speech and one voice is not split into two "speakers" because the background changed. The words themselves are still recognised from the original audio, because that measured more accurate than recognising from the cleaned copy. Mains hum and rumble below 70 Hz are removed either way.
3. **Stray voices are folded in.** A "speaker" with only a few seconds of audio in total (a cough, a door, one word in an odd tone) is merged into the real speaker it sounds most like instead of becoming Speaker 7.
4. **Every piece of text is checked.** Known phantom phrases and subtitle credits (in a dozen languages), `[Music]`-style markers, looped phrases, text with no real words, far more or far less text than the audio could hold, and text the model itself was unsure of are left out. With Whisper the language is decided once per file from the longest stretches of speech, so a noisy fragment cannot come back as a stray line of another language.

Nothing is hidden. The **Removed** button under a transcript lists every piece that was left out and why, and the JSON export carries the same list. The setting has three levels: **Balanced** (default), **Strict** (leaves out more, for poor recordings) and **Off** (keeps everything the model returned).

## Speakers: what to expect

Told the number of people, it assigns lines to that many voices and is right most of the time. Left to work it out, it can be one out, usually one too many when someone's voice changes a lot between turns. Either way, **Rename speakers…** fixes it in seconds: type real names, and give two speakers the same name to merge them. The saved files are rewritten with the new names.

Expect mistakes where people talk over each other, on very short turns ("yes", "okay"), and between voices that sound alike. Speaker labels are a strong first draft, not a record.

## Command line

The window is the main way in, but everything is available from a terminal:

```
verbatim transcribe meeting.m4a interview.mp3 --formats txt,docx --speakers 3
verbatim transcribe lecture.mp4 --model whisper-turbo --language cy --no-speakers -o ~/Desktop
verbatim models                 list models and which are installed
verbatim fetch whisper-small    download a model
verbatim selftest               check the whole pipeline
```

From source that is `python verbatim.py …` (or `run.bat …` / `./run.command …`); in a build it is `Verbatim.exe …` or `Verbatim.app/Contents/MacOS/Verbatim …`.

## How it works

```
decode (FFmpeg) -> find speech (Silero VAD) -> clean if noisy (GTCRN)
   -> who spoke when (pyannote segmentation + TitaNet voice fingerprints)
   -> transcribe each stretch (Parakeet or Whisper) -> check the text -> export
```

All models are ONNX files run by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on ONNX Runtime, on the CPU. No PyTorch, no GPU needed. As a guide, a two-core cloud machine transcribed with speaker labels at about four times real time (an hour of audio in roughly fifteen minutes); a recent laptop is quicker.

Each file is transcribed in a separate engine process, so Stop is instant, the window never freezes, and a problem with one file cannot take the app down.

| File | What it is |
|---|---|
| `verbatim.py` | Run from source |
| `verbatim_app.py` | Entry point for the built app |
| `verbatimlib/engine.py` | The pipeline |
| `verbatimlib/filters.py` | The invented-text checks |
| `verbatimlib/audio.py` | Decoding through FFmpeg, noise measurement |
| `verbatimlib/models.py` | Model catalogue and downloads |
| `verbatimlib/export.py` | The seven output formats |
| `verbatimlib/ui.py` | The window |
| `verbatimlib/cli.py` | Command line |
| `verbatimlib/selftest.py` | The self-test |
| `verbatimlib/theme.py` | Shared light/dark theme (same file as the other apps) |
| `ensure_python.ps1` | Finds or installs Python on Windows |

Settings (`verbatim-settings.json`), downloaded models and any crash log live beside the app, never inside it.

## Limits

- CPU only. Long recordings take a while; an hour of audio is minutes, not seconds.
- Speaker labels are less reliable with overlapping speech and similar voices (see above).
- Parakeet v2 is English only and Parakeet v3 has no Welsh; use Whisper for Welsh.
- A recording is decoded to memory-mapped temporary files, so a multi-hour file needs a few hundred MB of free disk while it runs.
- A transcript is a draft. Check names, numbers and anything that matters against the recording.

## Credits and licences

Verbatim itself is MIT-0 (see `LICENSE`). It stands on: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0), ONNX Runtime (MIT), FFmpeg via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), NVIDIA Parakeet TDT 0.6B (CC-BY-4.0), OpenAI Whisper (MIT), Silero VAD (MIT), pyannote segmentation 3.0 (MIT), NVIDIA TitaNet (CC-BY-4.0) and GTCRN (MIT).
