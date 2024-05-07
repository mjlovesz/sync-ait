
def get_entry_points(entry_points_name):
    try:
        from importlib import metadata

        return metadata.entry_points().get(entry_points_name, [])
    except:
        import pkg_resources

        return list(pkg_resources.iter_entry_points(entry_points_name))