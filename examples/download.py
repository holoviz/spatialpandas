try:
    import requests
    from platformdirs import user_cache_path
except ImportError:
    raise ImportError("requests and platformdirs are needed to download data")


def download_map(dataset):
    if dataset not in ("naturalearth_lowres", "naturalearth_cities"):
        raise ValueError(f"Unknown dataset: {dataset}, supported datasets are 'naturalearth_lowres' and 'naturalearth_cities'")
    url = f"https://api.github.com/repos/geopandas/geopandas/contents/geopandas/datasets/{dataset}?ref=v0.14.4"
    local_dir = user_cache_path() / "spatialpandas" / dataset

    if local_dir.exists():
        return local_dir

    response = requests.get(url)
    if response.status_code == 200:
        files = response.json()
    else:
        print(f"Failed to retrieve contents: {response.status_code}")
        return None

    if not local_dir.exists():
        local_dir.mkdir(parents=True)

    for file in files:
        file_url = file["download_url"]
        file_name = file["name"]
        file_response = requests.get(file_url)
        with open(local_dir / file_name, "wb") as f:
            f.write(file_response.content)

    return local_dir
