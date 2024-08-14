import os
from shutil import rmtree

try:
    import requests
    from platformdirs import user_cache_path
except ImportError:
    raise ImportError("requests and platformdirs are needed to download data") from None


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
    download_map("naturalearth_lowres")
    download_map("naturalearth_cities")
