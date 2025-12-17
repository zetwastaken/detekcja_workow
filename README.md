# detekcja_workow
Wykrywanie worków

## Konwencje jakości kodu
- Zainstaluj narzędzia deweloperskie: `pip install black pylint`.
- Uruchom `black .` aby sformatować kod.
- Uruchom `pylint --rcfile=.pylintrc` aby sprawdzić kod.
- W repozytorium działa workflow GitHub Actions `Quality`, który uruchamia te same kontrole przy każdym pushu i pull requeście.
