# Customer_service_calls_analyst

This project transcribes customer-service call audio, groups transcript segments by speaker using pitch features, and generates a structured call summary.

## Requirements

- Python 3.10 or later
- The packages listed in `requirements.txt`
- An internet connection the first time you run the project so Whisper and the summary model can be downloaded

## Setup on Windows

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run on a sample audio file

Audio samples are available in the `AxonData_samples` folder. For example, to analyze `1735404531.458927.mp3`, run this from the project folder:

```powershell
python Customer_service_analyst.py --audio "AxonData_samples/1735404531.458927.mp3" --call-id "0001" --whisper-model small
```
Another included sample is `AxonData_samples/1755884171.51632.mp3`.

## Output

By default, the script writes two files to `outputs`:

- `<call-id>_transcript.txt` — transcript segments with timestamps and speaker labels
- `<call-id>_summary.json` — structured summary and segment details

For the example command, the files are `outputs/0001_transcript.txt` and `outputs/0001_summary.json`. The output folder is created automatically if it does not exist.

## Command-line options

- `--audio` (required): audio file path, such as an MP3 or WAV file
- `--call-id`: output filename prefix; defaults to `call_001`
- `--whisper-model`: Whisper model (`tiny`, `base`, `small`, `medium`, or `large`); defaults to `base`
- `--summary-model`: Hugging Face summarization model; defaults to `facebook/bart-large-cnn`
- `--language`: audio language; defaults to `English`
- `--output-dir`: output folder; defaults to `outputs`
