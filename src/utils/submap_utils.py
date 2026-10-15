"""Dependency-free finalized-submap lookup helpers."""


def find_submap(frame_id: int, submaps: list):
    """Return the unique submap whose saved inclusive bounds contain a frame."""
    matches = [
        submap for submap in submaps
        if int(submap["submap_start_frame_id"]) <= int(frame_id)
        <= int(submap["submap_end_frame_id"])
    ]
    if len(matches) > 1:
        raise ValueError(f"frame {frame_id} belongs to overlapping submaps")
    return matches[0] if matches else None
