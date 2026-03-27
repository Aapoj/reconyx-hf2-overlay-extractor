# Reconyx HyperFire 2 Overlay Extractor

Tool to extract timestamp and temperature from overlay text using template matching.

This package is scoped to **Reconyx HyperFire 2 Cellular** overlays that match the included calibration artifacts.

## Included files
- `reconyx_hf2_extractor.py` - extraction engine
- `positions.json` - fixed timestamp character slot locations
- `glyph_crops/` - glyph templates (digits and separators)
- `requirements.txt` - runtime dependencies


## Setup
```powershell
python -m pip install -r requirements.txt
```

## Usage
Run extraction on a folder of images:

```powershell
python reconyx_hf2_extractor.py --input "PATH_TO_IMAGES" --positions positions.json --glyphs glyph_crops --csv outputs.csv --minutes-seconds-zero
```

Rename files to `RYYMMDDHH0000.JPG` while extracting:

```powershell
python reconyx_hf2_extractor.py --input "PATH_TO_IMAGES" --positions positions.json --glyphs glyph_crops --csv outputs.csv --minutes-seconds-zero --rename
```

### Rename format
When `--rename` is enabled, files are renamed to:
- Non-negative: `RYYYYMMDDHH0000pTT.JPG` (e.g. 4°C -> `...0000p04.JPG`, 0°C -> `...0000p00.JPG`)
- Negative: `RYYYYMMDDHH0000nTT.JPG` (e.g. -9°C -> `...0000n09.JPG`)

Where `TT` is always two digits.

### GUI mode
Launch the batch GUI (add multiple jobs, run in parallel, per-job + overall progress):

```powershell
python reconyx_hf2_extractor.py --gui
```

### Help
The script supports `--help` / `-h`:

```powershell
python reconyx_hf2_extractor.py --help
```

## Output CSV columns
- `original_file`
- `new_file`
- `timestamp`
- `timestamp_normalized`
- `temp_c`
- `ts_worst_score`
- `temp_worst_score`
- `status`
- `error`

## Limitations
- Calibrated for one overlay style (Reconyx HyperFire 2 Cellular).
- If overlay position/font differs, new `positions.json` and `glyph_crops/` are required.
- Minutes and Seconds are defaulted to zero in renaming.

