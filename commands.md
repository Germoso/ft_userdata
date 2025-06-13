### BACKTESTING

```jsx
freqtrade backtesting --strategy SampleStrategy --timeframe 5m --timerange 20240101-20250301

docker compose run --rm freqtrade backtesting --strategy RSIShortStrategy --timeframe 5m --timerange 20250101-20250601

freqtrade backtesting --strategy TrendFollowingStrategy --timeframe 5m --timerange 20250101-20250601

freqtrade backtesting --strategy RandomEntryStrategy --timeframe 5m --timerange 20250101-20250601
```

### BACKTESTING UI

```jsx
docker compose run -d --name freqtrade_bt_ui -p 127.0.0.1:8080:8080 freqtrade webserver

```

### DOWNLOAD DATA

```jsx
freqtrade download-data --days 365 --timeframes 1m 5m 15m 1h 4h 1d
docker compose run --rm freqtrade download-data --timeframe 5m --timerange 20240101-20250301
// Descargar todos los pares
docker compose run --rm freqtrade download-data --exchange binance --pairs ".*/USDT" --timeframe 5m --timerange 20240101-20250101
```

### HYPEROPT

```jsx
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy HourBasedStrategy -e 200 --timerange 20241201-20241231 --timeframe 1h

docker compose run --rm freqtrade hyperopt --hyperopt-loss ProfitDrawDownHyperOptLoss --strategy RSIShortStrategy -e 50 --timerange 20250101-20250301 --timeframe 5m

 

// MOSTRAR TODOS LOS RESULTADOS DE UN HYPEROPT RESULT
docker compose run --rm freqtrade hyperopt-show
```

### STRATEGY

```jsx
docker compose run --rm freqtrade new-strategy --template minimal --strategy LCD
```

### JUPYTER NOTEBOOK

```jsx
docker compose -f docker/docker-compose-jupyter.yml up
```