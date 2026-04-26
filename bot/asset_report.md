# Informe breve del bot por asset

## Resumen

El bot usa la misma lógica base en todos los símbolos de Binance Futures demo:

- vela de `5m`
- entrada por confluencia de `EMA 9/21/200`, `RSI`, `ADX`, `Supertrend` y volumen
- filtro de `ML` cuando ya hay historial suficiente
- gestión de riesgo con `break-even` y `trailing stop`
- ejecución solo si la posición local y la de Binance están alineadas

## Qué hace en cada asset

### `BTCUSDT`
- Busca tendencias limpias con buena liquidez.
- Es el activo de referencia para validar la estrategia base.

### `ETHUSDT`
- Aplica la misma lógica que BTC, pero suele moverse con algo más de beta.
- Sirve para capturar tendencias con menos ruido que altcoins pequeñas.

### `XRPUSDT`
- Requiere confluencia más clara porque puede romper y volver rápido.
- Útil para medir sensibilidad del filtro de volumen y del `ADX`.

### `SOLUSDT`
- Tiende a dar más impulsos y también más retrocesos bruscos.
- Es el asset donde más fácil se ve si el `trailing` está protegiendo bien.

### `XAUUSDT`
- Se trata como activo TradFi: más conservador y más sensible a tendencias limpias.
- Interesa revisar si el timeframe de `5m` es suficiente o si conviene subirlo.

### `XAGUSDT`
- Similar a XAU, pero normalmente más rápido y más errático.
- Buen candidato para feedback sobre filtros de ruido y falsas entradas.

## Puntos para revisar

- si el filtro de volumen está demasiado estricto
- si `ADX` y `RSI` están dejando pocos trades
- si `XAU/XAG` deberían usar parámetros distintos
- si el ML aporta mejora real antes de filtrar entradas
- si el `break-even` y el `trailing stop` están cerrando bien las posiciones

