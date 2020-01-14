First, retrieve and process the KEGG pathways using the `parse_KEGG.py` script.
Before using this script please ensure that you are authorized to access the
KEGG databse. More information can be found on:
[https://www.kegg.jp/kegg/legal.html](https://www.kegg.jp/kegg/legal.html)

To optimize experimental design for these pathways under various strategies use `run_KEGG.py`.
Please adjust the number of used cores and the maximum network size. See the module's docstring.

Finally, run `evaluate_KEGG_run.py` to process and visualize the results.
