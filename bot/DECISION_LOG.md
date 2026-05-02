# Decisión de portafolio: BTC observation-only

## Qué probamos
- Meta-labeling con triple barrier.
- Purged walk-forward con holdout sellado.
- Búsqueda iterativa de configuración durante 18 iteraciones.
- Hasta 5 años de data histórica BTC.
- Features técnicas, de flujo y de volumen.

## Qué no funcionó
- El modelo ML no replicó edge fuera de muestra.
- El `live_gate.json` quedó sellado con `ready_for_live=false` y `exhausted=true`.
- En BTC 1h, el mercado se comportó lo bastante eficiente como para arbitrar gran parte del edge aparente.
- Seguir optimizando thresholds o gates desde aquí sería p-hacking.

## Decisión tomada
- El ML queda degradado a observación.
- El bot opera un solo setup:
  - `vol_meanrev`
  - BTC-only
  - 1h
  - riesgo fijo `0.15%` del balance por trade
- No se abre un segundo setup hasta que este pase 90 días.

## Setup activo
- Overshoot mean reversion por volatilidad:
  - `return_1h_zscore_50 <= -2.5` + `RSI <= 25` + `ATR z >= 1.5` + `close > 0.85 * EMA200` => LONG
  - espejo para SHORT
- `TP = 0.6 * ATR`
- `SL = 1.0 * ATR`
- `time stop = 12 velas`

## Kill criterion
- Ventana de observación: `90 días`
- Si al cumplirla ocurre cualquiera de estas:
  - `sharpe < 0.3`
  - `profit_factor < 1.1`
  - `total_trades < 30`
  - `expectancy_R < 0`
- Entonces se escribe `KILL.flag` y el bot deja de abrir trades.

## Qué NO hacer sin reabrir esta decisión
- No volver a meter el ML en el path crítico.
- No bajar umbrales del kill criterion.
- No añadir un segundo setup antes de que este sobreviva 90 días.

## Si dentro de 6 meses esto falla
- Mirar primero `kill_criterion_log.json`, `KILL.flag` y `live_gate.json`.
- Si el setup no pasó, rediseñar desde data de derivados:
  - funding rate
  - open interest
  - basis
  - liquidations
  - flow real
