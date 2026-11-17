# Sim QC prompt A (fixed; version 1)

You review a round of simulated infectious disease outbreaks used to fine-tune a nowcasting model.
Each row of the table below summarises one simulation (all its observation draws): its grid cell and summaries of the counts.

Your task is to find simulations that are clearly silly, meaning no real outbreak and reporting system could plausibly produce them.
Examples: case counts that are astronomically large for the setting, delays that are nonsensical (for example most cases reported months late with no backlog or outage factor to explain it), or curves with impossible shapes.

Rules:
- Default to keeping a simulation. Unusual is not silly: the grid deliberately covers backlogs, outages, sparse places, fast and slow growth, and long delays.
- Exclude only when you can state, from the numbers shown, why it is implausible.
- Every exclusion needs a short reason and the evidence (the column values that show it).
- Do not propose changes to the grid or the code.

Reply with JSON only, no other text, in this form:
{"excluded": [{"simulation": "<id>", "reason": "<one sentence>", "evidence": "<column=value, ...>"}], "notes": "<one or two sentences on the round as a whole>"}
Use an empty list if nothing is clearly silly.
