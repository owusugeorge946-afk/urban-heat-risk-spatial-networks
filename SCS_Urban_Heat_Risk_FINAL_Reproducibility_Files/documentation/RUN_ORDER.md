# Final run order

1. Open the existing Earth Engine script **GeoAI-Based Modelling of Urban Heat Risk Propagation**. Confirm the 20 Imports are still visible.
2. Make a duplicate/copy of that GEE script before editing.
3. In the copy, preserve the Imports and paste the contents of `code/01_GEE_FINAL_200_TREES.js`.
4. Set `RUN_CITY = 'Accra'`; Run; inspect Console; start exports.
5. Repeat with Lagos, Abidjan and Freetown.
6. Keep all four `*_heat_network_nodes_500m_2024.geojson` files in `MyDrive/URBAN_HEAT_NETWORK_2024/`.
7. In Colab install packages from `requirements.txt`.
8. Run `code/02_COLAB_NETWORK_ANALYSIS.py`.
9. Inspect `REPRODUCIBILITY_RESULTS/MANUSCRIPT_BENCHMARK_AUDIT.csv`.
10. Archive the GEE console validation outputs and the Colab result tables with the Zenodo deposit.
