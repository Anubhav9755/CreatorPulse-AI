# services/youtube_client.py

import os
import re
from typing import List, Dict

import requests
import pandas as pd


YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeAPIError(Exception):
    pass


def _request(endpoint: str, params: Dict) -> Dict:
    if not YOUTUBE_API_KEY:
        raise YouTubeAPIError("YOUTUBE_API_KEY environment variable is not set.")

    merged = {"key": YOUTUBE_API_KEY, **params}
    r = requests.get(f"{YOUTUBE_API_BASE}/{endpoint}", params=merged, timeout=10)
    if r.status_code != 200:
        raise YouTubeAPIError(f"API error {r.status_code}: {r.text}")
    return r.json()


def extract_channel_id_from_url(url_or_handle: str) -> str:
    """Accept full URL or @handle and resolve to channelId."""
    text = url_or_handle.strip()

    # Handle direct channel URLs
    m = re.search(r"channel/([A-Za-z0-9_-]+)", text)
    if m:
        return m.group(1)

    # Handle '@handle'
    if text.startswith("@") or "youtube.com/" in text:
        # Use search endpoint to resolve handle or custom URLs
        resp = _request(
            "search",
            {
                "part": "snippet",
                "q": text.replace("https://", "").replace("http://", ""),
                "type": "channel",
                "maxResults": 1,
            },
        )
        items = resp.get("items", [])
        if not items:
            raise YouTubeAPIError("Could not resolve channel from URL or handle.")
        return items[0]["snippet"]["channelId"]

    raise YouTubeAPIError("Input is not a valid YouTube channel link or handle.")


def get_uploads_playlist_id(channel_id: str) -> str:
    resp = _request(
        "channels",
        {
            "part": "contentDetails",
            "id": channel_id,
        },
    )
    items = resp.get("items", [])
    if not items:
        raise YouTubeAPIError("Channel not found.")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_all_videos_from_playlist(playlist_id: str) -> List[Dict]:
    """Return list of playlistItem dicts (videoId + snippet)."""
    items = []
    page_token = None
    while True:
        resp = _request(
            "playlistItems",
            {
                "part": "snippet,contentDetails",
                "playlistId": playlist_id,
                "maxResults": 50,
                **({"pageToken": page_token} if page_token else {}),
            },
        )
        items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def fetch_video_stats(video_ids: List[str]) -> Dict[str, Dict]:
    """Return mapping videoId -> stats dict."""
    stats_map: Dict[str, Dict] = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = _request(
            "videos",
            {
                "part": "statistics,contentDetails,snippet",
                "id": ",".join(batch),
                "maxResults": 50,
            },
        )
        for item in resp.get("items", []):
            vid = item["id"]
            stats_map[vid] = {
                "title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
                "comment_count": int(item["statistics"].get("commentCount", 0)),
                "duration_iso": item["contentDetails"]["duration"],
            }
    return stats_map


def iso8601_duration_to_seconds(duration: str) -> int:
    """Convert ISO 8601 duration (e.g., PT12M5S) to seconds."""
    pattern = re.compile(
        r"P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?"
    )
    match = pattern.fullmatch(duration)
    if not match:
        return 0
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def build_channel_videos_dataframe(channel_url_or_handle: str) -> pd.DataFrame:
    """End‑to‑end: channel link -> DataFrame with core metrics."""
    channel_id = extract_channel_id_from_url(channel_url_or_handle)
    uploads_playlist = get_uploads_playlist_id(channel_id)
    playlist_items = fetch_all_videos_from_playlist(uploads_playlist)

    video_ids = [it["contentDetails"]["videoId"] for it in playlist_items]
    stats_map = fetch_video_stats(video_ids)

    rows = []
    for it in playlist_items:
        vid = it["contentDetails"]["videoId"]
        stat = stats_map.get(vid)
        if not stat:
            continue
        published = pd.to_datetime(stat["published_at"], errors="coerce")
        rows.append(
            {
                "video_id": vid,
                "title": stat["title"],
                "published_at": published,
                "upload_hour": published.hour if not pd.isna(published) else None,
                "day_of_week": published.dayofweek if not pd.isna(published) else None,
                "views_24h": stat["view_count"],  # rough proxy (public total views)
                "views_7d": stat["view_count"],   # you can refine later
                "like_count": stat["like_count"],
                "comment_count": stat["comment_count"],
                "video_duration_sec": iso8601_duration_to_seconds(stat["duration_iso"]),
            }
        )

    df = pd.DataFrame(rows)
    return df
