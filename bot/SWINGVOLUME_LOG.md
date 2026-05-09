# SWINGVOLUME

## Decisión

Bot anterior no mostró edge robusto en BTC 1h:

- meta-labeling con 57 features
- 18 iteraciones de búsqueda
- hasta 5 años de data
- hold-out sin réplica

Decisión: sacar ML de path crítico. Operar un solo setup discrecional cuantificado. Si no pasa 90 días demo, apagar.

## Setup

Timeframes:

- contexto macro: `D1`
- trigger: `H4`

Sesgo D1:

- `EMA20_D1 > EMA200_D1` -> solo `LONG`
- `EMA20_D1 < EMA200_D1` -> solo `SHORT`
- cierre D1 no puede quedar en extremo de la vela:
  - `close_pos = (close-low)/(high-low)`
  - requiere `0.02 <= close_pos <= 0.98`

Divergencia MACD H4:

- usar histograma `MACD(12,26,9)`
- normalizar como `% del precio`: `macd_hist_pct = macd_hist / close`
- para `LONG`:
  - primer bajo profundo `<= -0.0005`
  - segundo bajo entre 2 y 8 velas después
  - precio hace `lower low`
  - histograma hace `higher low`
- para `SHORT`: espejo
- divergencia expira en `4` velas H4

Vela de volumen H4:

- `volume > vol_ma20 * 1.3`
- `vol_zscore_20 >= 1.5`
- dos velas previas con `volume < vol_ma20 * 0.9`
- cuerpo `>= 60%` del rango
- vela direccional:
  - `LONG`: `close > open`
  - `SHORT`: `close < open`

Recuperación MACD:

- `LONG`: `-0.0001 <= macd_hist_pct <= 0.0005`
- o cruce de negativo a positivo
- `SHORT`: espejo alrededor de cero

## Ejecución

Entrada:

- al cierre de vela H4 que confirma todo

Stop:

- `open` de vela gatillo
- si no sirve, fallback `1.2 * ATR(14)`

Target:

- `TP = 3R`

Time stop:

- `5` velas H4

Trailing:

- `+0.3R` -> stop a breakeven
- `+0.7R` -> stop a `+0.2R`

## Riesgo

- `0.15%` del balance por trade
- máximo `1` trade abierto
- máximo `1` trade nuevo por día UTC
- daily cap `0.5%`
- streak `3` pérdidas -> pausa `48h`
- rolling `7` pérdidas en `20` trades (`win rate < 35%`) -> pausa `7d`

## Kill criterion

En día `90`:

- `sharpe >= 0.4`
- `profit_factor >= 1.15`
- `win_rate >= 50%`
- `total_trades >= 15`
- `max_drawdown < 30%`

Si falla cualquiera:

- escribir `KILL.flag`
- no abrir más trades

## Regla de gobierno

No reintroducir ML al path crítico sin actualizar esta nota y justificarlo con edge fuera de muestra real.
