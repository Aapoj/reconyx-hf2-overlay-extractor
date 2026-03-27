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

### GUI mode
Launch a simple GUI (folder picker + progress bar):

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

