## Environment
- Python conda environment to use: `../conda_env_gzh_data0.yml` (needs `requests`, `beautifulsoup4`, `pandas`, `pyarrow`)
- Docker compose setup: none
- Environment variables: none

## How to run
Run the scripts from the repository root. Both scripts accept `--site auto|remote|local`:
`auto` (default) scrapes the real sandbox site (https://quotes.toscrape.com, https://books.toscrape.com) if it is
reachable and otherwise starts a local sandbox web site with the same HTML structure, so they also run offline.
- `python -m p02_data_sourcing.c04_scraping_data.main_1_scrape_quotes` - robots.txt check, pagination,
  parsing quotes into a DataFrame, CSV/Parquet export. Options: `--max-pages 3 --delay 1.0`.
- `python -m p02_data_sourcing.c04_scraping_data.main_2_scrape_books` - listing and detail pages, HTML
  tables, value cleaning, joining both levels. Options: `--max-pages 2 --n-details 6 --delay 1.0`.

Please keep `--delay` at 1 second or more when scraping real web sites, and read the legal and ethical
section in `book_scraping_data.md` before scraping anything else.
Results are stored in `p02_data_sourcing/c04_scraping_data/outputs/` (git-ignored).
