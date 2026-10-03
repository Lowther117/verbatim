# Verbatim

Turn any audio or video file of English speech into text, with each speaker labelled. Queue up as many recordings as you like and it works through them one after another. Everything runs on your own computer: no audio, text or file name is sent anywhere, and there is no account, key or subscription.

Windows and macOS.

## What it does

- **Any format in.** MP3, M4A, WAV, FLAC, OGG, Opus, WMA, AAC, AMR, AIFF, and the sound track of MP4, MKV, MOV, WebM, AVI and the rest. A static FFmpeg is built in, so nothing needs installing.
- **A queue.** Add files, add a whole folder, or drop them on the window. Add more while it is running. One failed file does not stop the rest.
- **Who said what.** Each voice becomes Speaker 1, Speaker 2 and so on. Tell it how many people there are if you know, rename them afterwards, give two speakers the same name to merge them, and right-click any paragraph to say who really spoke it.
- **Noise handling and invented-text protection.** See [Keeping invented text out](#keeping-invented-text-out).
- **Seven outputs.** Plain text, Word (.docx), Markdown, SRT and VTT subtitles, CSV, and JSON with every timestamp and confidence score. Tick the ones you want and each transcript is saved in all of them.
- **Saved to Downloads.** Transcripts go to `Downloads/Verbatim` unless you choose another folder (File > Default save folder…). An existing file is never overwritten: the new one becomes `name (2).txt`.
- **Come back later.** Tick JSON and the saved file can be reopened (File > Open saved transcript) to rename speakers or save other formats without transcribing again.
- **Looks after the machine.** While a file is running the computer is kept awake, and the work runs at below-normal priority so everything else stays usable. Stop is immediate and leaves nothing behind.

## Which file do I use?

| I want to… | Windows | macOS |
|---|---|---|
| Build a standalone app (recommended) | `build-exe.bat` | `build-app.command` |
| Run from source without building | `run.bat` | `run.command` |

Both build files are double-click and install everything themselves: Python if it is missing (Homebrew too, on a Mac), every dependency, and the speech models. The first build downloads about 500 MB of models; they are kept in `build-models/` so later builds do not download them again, and only the ones chosen for a build are baked into it.

- **Windows** produces the folder `dist\Verbatim` with `Verbatim.exe` inside. It is a folder rather than a single exe on purpose: roughly 700 MB of models are baked in, and a one-file exe would unpack all of it to a temporary folder at every launch. Move the whole folder wherever you like.
- **macOS** produces `dist/Verbatim.app`. Drag it to Applications.

Each build ends by running the app's self-test, which transcribes a short two-speaker sample for real, feeds it 20 seconds of noise and checks nothing is "transcribed", round-trips every export format, runs the engine as the window does and builds the window. It says `ALL GOOD` or `PROBLEMS FOUND`; the report is `verbatim-selftest.txt` next to the app.

On a Mac, a new `.command` file may need this once in Terminal before it will double-click:

```
cd ~/Downloads/GitHub/verbatim
chmod +x build-app.command run.command
xattr -d com.apple.quarantine build-app.command run.command 2>/dev/null
```

To bake in a different speech model, set `VERBATIM_BAKE` before building (`whisper-turbo`, `whisper-small`, or `none` to bake only the small helper models and let the app download a speech model on first use).

## Speech models

English only. Three models, and for nearly everything the first is the one to use.

| Model | Size | Speed | Use it for |
|---|---|---|---|
| **Parakeet v2** (default, baked in) | 630 MB | fast | Everything. The most accurate of the three and the least likely to invent words. It times every word, which is what lets a sentence be shared out cleanly between two speakers. |
| Whisper large-v3 turbo | 1 GB | slow | A second opinion on a recording Parakeet struggled with: heavy accents, unusual names. More inclined to invent text over noise. |
| Whisper small | 360 MB | medium | A lighter second opinion on a computer short of memory. |

The Whisper models are multilingual but are held to English here. Left to detect the language, Whisper guesses afresh for every piece of audio, and on a short or noisy piece the guess can be anything, which is how a stray line of another language ends up in an English meeting.

Models other than the baked-in one are downloaded once from **Models > Download and manage models…**. They come from the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project's GitHub releases; nothing is fetched from anywhere else and no token is needed.

## Keeping invented text out

Speech models, Whisper especially, were trained on subtitled video. Hand them silence, music or hiss and they answer with what subtitles say at such moments: "Thanks for watching", a credit line, a lone "you", or one phrase looped forty times. None of it was said. Verbatim has four defences, in this order:

1. **Only speech is transcribed.** A speech detector (Silero VAD) marks where someone is actually talking, and only those stretches ever reach the speech model. Silence, music and background noise are never shown to it.
2. **Noise reduction when it is needed.** The background level is measured from the gaps between speech. If the recording is noisy, the file goes through a noise-reduction model (GTCRN) and speech is detected again on the cleaned audio, so noise bursts are not mistaken for speech and one voice is not split into two "speakers" because the background changed. The words themselves are still recognised from the original audio, because that measured more accurate than recognising from the cleaned copy. Mains hum and rumble below 70 Hz are removed too, unless noise reduction is set to Off.
3. **Stray voices are folded in.** A "speaker" with only a few seconds of audio in total (a cough, a door, one word in an odd tone) is merged into the real speaker it sounds most like instead of becoming Speaker 7.
4. **Every piece of text is checked.** Known phantom phrases and subtitle credits, `[Music]`-style markers, looped phrases, text with no real words, far more or far less text than the audio could hold, and text the model itself was unsure of are left out. The checks are deliberately narrow: a genuine "No.", "Thank you" or "Bye-bye" in the flow of a conversation is kept, and the same words are dropped only when the evidence is weak (very quiet, low confidence, or alone with silence either side).

Nothing is hidden. The **Removed** button under a transcript lists every piece that was left out and why, and the JSON export carries the same list. The setting has three levels: **Balanced** (default), **Strict** (leaves out more, for poor recordings) and **Off** (keeps everything the model returned).

## Speakers: what to expect

Working out how many people are speaking is the hardest part. It is done in two passes: voices are first split into many small groups, then each group's voice is averaged over everything it said and groups that sound like the same person are merged. The merge threshold was set by measurement on 32 test recordings of real voices, including ones where the speaking level, the microphone and the background noise change between turns. Left to work it out, it found the right number of people in 27 of the 32 and was never out by more than one. (Version 1.0, tuned on synthetic voices, managed 14 of 32 and often reported four or more speakers for two.)

Told the number of people, it returns that many (provided there are that many distinct voices to find) and attributes more of the speech correctly.

Either way, mistakes are quick to fix afterwards:

- **Rename speakers…** puts real names in. Give two speakers the same name to merge them.
- **Right-click a paragraph** to hand it to a different speaker, or to someone new.

The saved files are rewritten each time. Expect mistakes where people talk over each other, on very short turns ("yes", "okay"), between voices that sound alike, and when one person's sound changes a lot mid-recording (moving from a headset to a phone, say), which can still show up as an extra speaker. Speaker labels are a strong first draft, not a record.

## Speed, and using the graphics processor

**Speed > Test this computer…** times the different ways of running on two minutes of sample audio and keeps whichever is quickest on that computer. It is worth doing once per machine, because none of this can be decided from a specification sheet:

- **One stage at a time, or both at once.** Working out the speakers and transcribing do not depend on each other, so with processor cores to spare they can run side by side. On a two-core machine it made no difference; on a machine with more it should help, and the test finds out.
- **Graphics processor.** On a **Mac** the build includes Apple's Core ML back end, and the test tries it. Whether it beats the processor for these particular models varies; it is used only if it is at least 15% quicker. On **Windows** the standard build is processor-only. For an **NVIDIA** card, build with `set VERBATIM_GPU=cuda` before running `build-exe.bat`: that swaps in the CUDA build of the speech engine and needs the CUDA 12 and cuDNN 9 runtimes installed on the PC. That path is experimental and has not been tested by the author; the speed test will only switch to it if it works and is quicker.

A back end that fails, or quietly falls back to the processor, is never chosen. The result is saved and shown under Speed.

## Command line

The window is the main way in, but everything is available from a terminal:

```
verbatim transcribe meeting.m4a interview.mp3 --formats txt,docx --speakers 3
verbatim transcribe lecture.mp4 --model whisper-turbo --no-speakers -o ~/Desktop
verbatim speedtest              time this computer and keep the quickest way to run
verbatim models                 list models and which are installed
verbatim fetch whisper-small    download a model
verbatim selftest               check the whole pipeline
```

From source that is `python verbatim.py …` (or `run.bat …` / `./run.command …`); in a build it is `Verbatim.exe …` from a command prompt, or `Verbatim.app/Contents/MacOS/Verbatim …`.

## How it works

```
decode (FFmpeg) -> find speech (Silero VAD) -> clean if noisy (GTCRN)
   -> who spoke when (pyannote segmentation + TitaNet voice fingerprints)
   -> transcribe each stretch (Parakeet or Whisper) -> check the text -> export
```

All models are ONNX files run by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) on ONNX Runtime. No PyTorch, and no graphics processor needed. As a guide, a two-core cloud machine transcribed with speaker labels at four to six times real time (an hour of audio in ten to fifteen minutes); a recent laptop is quicker.

Each file is transcribed in a separate engine process, so Stop is instant, the window never freezes, and a problem with one file cannot take the app down. The model is given whole stretches of speech, cut only at pauses, and each word is then given to whoever was speaking when it was said; cutting the audio at each change of speaker would clip the words either side of the cut.

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
| `verbatimlib/proc.py` | Starting and stopping the engine process |
| `verbatimlib/speedtest.py` | The speed test |
| `verbatimlib/selftest.py` | The self-test |
| `verbatimlib/theme.py` | Shared light/dark theme (same file as the other apps) |
| `ensure_python.ps1` | Finds or installs Python on Windows |

Settings (`verbatim-settings.json`), downloaded models and any crash log live beside the app, never inside it. If that folder cannot be written to (the app is in `/Applications` or `Program Files`), they go to the per-user application data folder instead; Help > About shows which.

## Limits

- English only.
- Long recordings take a while; an hour of audio is minutes, not seconds.
- Speaker labels are less reliable with overlapping speech and similar voices (see above).
- A recording is decoded to memory-mapped temporary files, so a multi-hour file needs a few hundred MB of free disk while it runs.
- A transcript is a draft. Check names, numbers and anything that matters against the recording.

## Credits and licences

Verbatim itself is MIT-0 (see `LICENSE`). It stands on: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0), ONNX Runtime (MIT), FFmpeg via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), NVIDIA Parakeet TDT 0.6B (CC-BY-4.0), OpenAI Whisper (MIT), Silero VAD (MIT), pyannote segmentation 3.0 (MIT), NVIDIA TitaNet (CC-BY-4.0) and GTCRN (MIT).
