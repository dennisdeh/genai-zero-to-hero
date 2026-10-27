"""
Helpers to check (never print!) which Kaggle credentials are available on this machine.

The Kaggle client looks for credentials in several places (see book_kaggle.md for details):
    - environment variable KAGGLE_API_TOKEN, or the file ~/.kaggle/access_token (new-style API tokens),
    - environment variables KAGGLE_USERNAME and KAGGLE_KEY (legacy API key),
    - the file kaggle.json in $KAGGLE_CONFIG_DIR, ~/.kaggle/ or (Linux) ~/.config/kaggle/ (legacy API key),
    - cached OAuth credentials (~/.kaggle/credentials.json) created by `kaggle auth login` (recent clients).
"""

import os


def kaggle_config_dirs() -> list[str]:
    """Return the directories in which the Kaggle client looks for its kaggle.json configuration file."""
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        return [os.environ["KAGGLE_CONFIG_DIR"]]
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return [os.path.join(os.path.expanduser("~"), ".kaggle"), os.path.join(xdg, "kaggle")]


def credential_sources() -> list[str]:
    """
    Return a list describing which credential sources are present. Only the *names* of the
    sources are returned; the secret values are never read into the output.
    """
    found = []
    home_kaggle = os.path.join(os.path.expanduser("~"), ".kaggle")
    if os.environ.get("KAGGLE_API_TOKEN"):
        found.append("environment variable KAGGLE_API_TOKEN")
    if os.path.isfile(os.path.join(home_kaggle, "access_token")):
        found.append("file ~/.kaggle/access_token")
    if os.path.isfile(os.path.join(home_kaggle, "credentials.json")):
        found.append("OAuth credentials ~/.kaggle/credentials.json (kaggle auth login)")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("environment variables KAGGLE_USERNAME + KAGGLE_KEY")
    for config_dir in kaggle_config_dirs():
        path_json = os.path.join(config_dir, "kaggle.json")
        if os.path.isfile(path_json):
            found.append(f"file {path_json}")
            if os.name == "posix" and (os.stat(path_json).st_mode & 0o077):
                print(f"Warning: {path_json} is readable by other users; run `chmod 600 {path_json}`")
    return found


HOW_TO_AUTHENTICATE = """
No usable Kaggle credentials were found. To use the Kaggle API:
  1. Create a free account on https://www.kaggle.com and open Settings -> API.
  2. Create an API token and provide it in ONE of these ways (never hard-code it in a script
     and never commit it to git):
       a) export KAGGLE_API_TOKEN=<your token>            (new-style token, environment variable)
       b) save the token in ~/.kaggle/access_token         (new-style token, file)
       c) put the legacy kaggle.json file in ~/.kaggle/ and run `chmod 600 ~/.kaggle/kaggle.json`
       d) export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>   (legacy key, e.g. in CI pipelines)
       e) run `kaggle auth login` for a browser-based OAuth login (recent client versions)
  3. For competitions you must additionally accept the competition rules on the website once.
"""
