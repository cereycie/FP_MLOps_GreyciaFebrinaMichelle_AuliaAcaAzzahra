# README Data

This folder does not include the raw dataset directly, since `events_scored.csv` is around 73 MB, above what GitHub's web upload allows.

To get the file:

1. It is produced during Hands-On 1, as an intermediate export of the cleaned Chicago Crimes event-level data with severity scores already computed.
2. If you have access to the shared Google Drive folder for this project, download `events_scored.csv` from there and place it in this `data/` folder before running any notebook.
3. Expected columns: `Datetime`, `cell_id`, `lat_r`, `lon_r`, `dow`, `hour`, `Primary Type`, `Description`, `severity`, `age_days`, `w_time`, `event_value`.
