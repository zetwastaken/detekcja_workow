# detekcja_workow
Wykrywanie worków

## Konwencje jakości kodu
- Zainstaluj narzędzia deweloperskie: `pip install pre-commit`.
- Aktywuj hooki lokalnie: `pre-commit install` dzięki czemu `black` sformatuje kod, a `pylint` go sprawdzi przed każdym commitem.
- W razie potrzeby uruchom ręcznie pełny pakiet kontroli: `pre-commit run --all-files`.
- W repozytorium działa workflow GitHub Actions `Quality`, który uruchamia te same kontrole przy każdym pushu i pull requeście.
