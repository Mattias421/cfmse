# Data-efficiency speaker splits

These are nested subsets of the 26 speakers in the existing 16 kHz VB+DMD
training set. They were produced by shuffling the sorted speaker IDs with Python's
`random.Random(1337)` and taking the first 26, 13, 7, 3, or 1 IDs. The files are
sorted for readability after selection.

`p226` and `p287` are deliberately absent: they are the validation speakers.
The standard test speakers (`p232` and `p257`) are also disjoint. Run
`python xps/validate_data.py --base-dir /path/to/VB+DMD` before launching jobs.

The exact medium/low/very-low sizes and selected single speaker were not fixed in
`xp_plan.md`; these are reproducible operational defaults and should be confirmed
before the full sweep.
