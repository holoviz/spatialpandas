import os
import sys
import time
from shutil import rmtree

try:
    import requests
    from platformdirs import user_cache_path
except ImportError:
    raise ImportError("requests and platformdirs are needed to download data") from None


def retry(func, *args, **kwargs):
    for i in range(5):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait = 10 * 2**i
            print(f"Attempt {i + 1} failed: {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return func(*args, **kwargs)


if os.environ.get("GITHUB_TOKEN"):
    HEADERS = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
else:
    HEADERS = None


def download_map(dataset):
    if dataset not in ("naturalearth_lowres", "naturalearth_cities"):
        raise ValueError(
            f"Unknown dataset: {dataset}, supported datasets are 'naturalearth_lowres' and 'naturalearth_cities'"
        )
    url = f"https://api.github.com/repos/geopandas/geopandas/contents/geopandas/datasets/{dataset}?ref=v0.14.4"
    local_dir = user_cache_path() / "spatialpandas" / dataset

    if local_dir.exists():
        return local_dir

    response = requests.get(url, headers=HEADERS)
    if response.ok:
        files = response.json()
    else:
        raise ValueError(
            f"Failed to retrieve contents ({response.status_code}): \n {response.text}"
        )

    if not local_dir.exists():
        local_dir.mkdir(parents=True)

    for file in files:
        file_url = file["download_url"]
        file_name = file["name"]
        file_response = requests.get(file_url, headers=HEADERS)
        if not file_response.ok:
            rmtree(local_dir)
            raise ValueError(f"Failed to download file: {file_name}, \n{file_response.text}")
        with open(local_dir / file_name, "wb") as f:
            f.write(file_response.content)

    return local_dir


if __name__ == "__main__":
    retry(download_map, "naturalearth_lowres")
    retry(download_map, "naturalearth_cities")
