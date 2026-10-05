# Reproducible figure-generation package for the current manuscript

## Scope

Figures 1-2 are controlled technical diagrams, not data plots. Their editable source is:
`esquemas_traducidos_ingles.pptx`.

The Python scripts in this package reproduce all data-based manuscript figures 3-14.

## Scripts

- `43_generar_figuras_03_08_EDA_CORREGIDO.py`
  - Figures 3-8 from the authoritative Bronze 1-minute and Gold hourly data.
  - Figure 3(c): linear x-axis limited to the 99th percentile of volume for display.
  - Figure 3(d): linear x-axis limited to the 99th percentile of trades for display.
  - Figure 5(a): symmetric x-axis limited to ±P99.9 of |one-minute log return| for display.
  - Full extreme observations remain in the source and in the manuscript tables.
- `44_generar_figuras_09_14_BENCHMARK_ACTUAL.py`
  - Figures 9-14 using the definitive scenario, multiseed, prediction and statistical outputs.
- `45_generar_todas_figuras_Q1.py`
  - Runs both generators and checks that Figures 3-14 were created.
- `46_reemplazar_figuras_3_5_en_manuscrito.py`
  - Replaces Figures 3 and 5 in a DOCX after local regeneration and updates their captions transparently.

## Expected source paths

Bronze:
`data\bronze\spot\BTCUSDT\1m\btcusdt_spot_1m_bronze_2019_2026.parquet`

Gold (first available is used):
`data\gold\spot\BTCUSDT\1h\btcusdt_spot_1h_gold_2019_2026.parquet`
or
`data\1h\global\btcusdt_spot_1h_model_input_2019_2026.parquet`

Definitive outputs used by Figures 9-14:
`outputs\scenarios_final_seed42\...`
`outputs\reproducibility\multiseed_final_global_resumen.csv`
`outputs\statistical_analysis\global_final_dm_hierarchical_bootstrap.csv`

## Recommended Windows PowerShell commands

Copy the four `.py` files into `src\`, then from the project root run:

```powershell
python src\45_generar_todas_figuras_Q1.py
```

The complete plot set is written to:

`outputs\figures_q1_final_complete\`

To insert the corrected Figure 3 and Figure 5 into the updated manuscript:

```powershell
python src\46_reemplazar_figuras_3_5_en_manuscrito.py `
  "LogachoA_Articulo_SPRINGER_NATURE_Q1_FINAL_M3_M5_MZ_TABLE5_MATH_FINAL.docx" `
  "LogachoA_Articulo_SPRINGER_NATURE_Q1_FINAL_COMPLETO_FIGURAS_CORREGIDAS.docx" `
  --figures-dir "outputs\figures_q1_final_complete"
```

The script also changes the two captions to disclose the display-only percentile axis limits.

## Important

The percentile limits affect visualization only. They do not delete source observations,
change model inputs, modify metrics, or remove extreme events from the descriptive tables.
