# data/synthetic_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse

RNG_SEED = 42


NICHES = [
    "gaming", "education", "vlog", "tech", "beauty", "finance",
    "music", "fitness", "food", "travel"
]

COUNTRIES = ["US", "IN", "GB", "CA", "DE", "BR", "AU"]

TOPICS = [
    "how to", "review", "tutorial", "reaction", "vlog", "challenge",
    "news", "explainer", "podcast", "shorts"
]


def generate_channels(n_channels: int, rng: np.random.Generator) -> pd.DataFrame:
    # Subscriber count: heavy-tailed (few very big, many small)
    # log10(subs) ~ Uniform(2, 7) => 100 to 10M
    subs_log = rng.uniform(2, 7, size=n_channels)
    subs = np.round(10 ** subs_log).astype(int)

    niches = rng.choice(NICHES, size=n_channels)
    countries = rng.choice(COUNTRIES, size=n_channels)

    # Channel-level "quality" factor affecting performance slightly
    quality = rng.normal(loc=1.0, scale=0.15, size=n_channels).clip(0.5, 1.5)

    channels = pd.DataFrame({
        "channel_id": [f"ch_{i:05d}" for i in range(n_channels)],
        "niche": niches,
        "country": countries,
        "subscriber_count": subs,
        "channel_quality": quality
    })

    return channels


def sample_publish_time(n: int, rng: np.random.Generator):
    # Random timestamps over last 365 days
    now = datetime.utcnow()
    days_back = rng.integers(0, 365, size=n)
    hours = rng.integers(0, 24, size=n)
    minutes = rng.integers(0, 60, size=n)

    timestamps = []
    for d, h, m in zip(days_back, hours, minutes):
        ts = now - timedelta(days=int(d), hours=int(h), minutes=int(m))
        timestamps.append(ts.replace(second=0, microsecond=0))
    return np.array(timestamps)


def generate_video_titles(n: int, topics: np.ndarray, niches: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    # Very simple synthetic titles
    templates = [
        "Best {topic} for {niche}",
        "Top 10 {niche} {topic}",
        "I tried {topic} in {niche}",
        "{topic} changed my {niche} channel",
        "Do this for better {niche} {topic}",
        "Stop doing this in {niche} {topic}"
    ]
    titles = []
    for i in range(n):
        t = rng.choice(templates)
        title = t.format(topic=topics[i], niche=niches[i])
        titles.append(title)
    return np.array(titles)


def generate_videos(channels: pd.DataFrame,
                    n_videos_total: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    n_channels = len(channels)

    # Decide how many videos per channel (skewed: some channels post more)
    # Use Poisson around 50, but min 10
    vids_per_channel = rng.poisson(lam=50, size=n_channels) + 10
    vids_per_channel = (vids_per_channel / vids_per_channel.sum() * n_videos_total).astype(int)
    # Adjust to exact total
    diff = n_videos_total - vids_per_channel.sum()
    if diff != 0:
        vids_per_channel[0] += diff

    rows = []
    video_id_counter = 0

    for ch_idx, ch in channels.iterrows():
        n_v = vids_per_channel[ch_idx]
        if n_v <= 0:
            continue

        channel_id = ch["channel_id"]
        subs = ch["subscriber_count"]
        niche = ch["niche"]
        country = ch["country"]
        channel_quality = ch["channel_quality"]

        # Video-level attributes
        published_at = sample_publish_time(n_v, rng)
        day_of_week = np.array([ts.weekday() for ts in published_at])  # 0=Mon
        upload_hour = np.array([ts.hour for ts in published_at])

        # Topic & durations
        topics = rng.choice(TOPICS, size=n_v)
        durations = rng.gamma(shape=2.0, scale=300.0, size=n_v)  # mean ~600s
        durations = durations.clip(60, 3600).astype(int)

        # Title & thumbnail text
        titles = generate_video_titles(n_v, topics, np.full(n_v, niche), rng)
        title_lengths = np.array([len(t) for t in titles])
        thumb_text_len = rng.integers(10, 60, size=n_v)

        # Sentiment (roughly positive)
        sentiment = rng.normal(loc=0.2, scale=0.4, size=n_v).clip(-1, 1)

        # Topic embeddings (small dense vector, e.g. 16-dim)
        emb_dim = 16
        topic_emb = rng.normal(size=(n_v, emb_dim)).astype(np.float32)

        # Base views using subs and some randomness
        # ~ base_views = subs * factor * random
        # smaller channels are more volatile, big channels more stable
        base_factor = rng.lognormal(mean=0, sigma=1.0, size=n_v)
        size_stability = np.log10(subs + 10)
        stability_factor = (1.5 / size_stability)  # smaller subs => higher
        base_views_24h = subs * base_factor * stability_factor * channel_quality

        # Time-of-day effect: 18-22h tends to perform better
        prime_hours = ((upload_hour >= 18) & (upload_hour <= 22)).astype(float)
        hour_boost = 1.0 + 0.4 * prime_hours

        # Day-of-week effect: weekends better
        weekend = ((day_of_week == 5) | (day_of_week == 6)).astype(float)
        dow_boost = 1.0 + 0.3 * weekend

        # Topic multipliers (e.g. shorts, challenges more viral)
        topic_base = np.ones(n_v)
        topic_base[topics == "shorts"] *= 1.6
        topic_base[topics == "challenge"] *= 1.4
        topic_base[topics == "news"] *= 1.2
        topic_base[topics == "podcast"] *= 0.8

        # Duration sweet-spot: 6-12 min (360-720s)
        duration_opt_center = 540
        duration_penalty = np.exp(-((durations - duration_opt_center) ** 2) / (2 * (300 ** 2)))

        # Combine
        views_24h = base_views_24h * hour_boost * dow_boost * topic_base * (0.7 + 0.6 * duration_penalty)
        # Add randomness and floor at 0
        views_24h = (views_24h * rng.lognormal(mean=0, sigma=0.8, size=n_v)).clip(0, None)

        # 7d views larger than 24h
        growth_factor_7d = rng.uniform(1.2, 4.0, size=n_v)
        views_7d = views_24h * growth_factor_7d

        # Engagement metrics
        # like_rate ~1-10%, comment_rate ~0.1-2%, shares smaller
        base_like_rate = rng.uniform(0.01, 0.1, size=n_v)
        base_comment_rate = rng.uniform(0.001, 0.02, size=n_v)
        base_share_rate = rng.uniform(0.0005, 0.01, size=n_v)

        # Sentiment affects like/comment positively
        sentiment_boost = 1.0 + 0.2 * sentiment

        likes_24h = (views_24h * base_like_rate * sentiment_boost).astype(int)
        comments_24h = (views_24h * base_comment_rate * sentiment_boost).astype(int)
        shares_24h = (views_24h * base_share_rate * sentiment_boost).astype(int)

        like_ratio_24h = np.divide(
            likes_24h, views_24h,
            out=np.zeros_like(likes_24h, dtype=float),
            where=views_24h > 0
        )
        comment_ratio_24h = np.divide(
            comments_24h, views_24h,
            out=np.zeros_like(comments_24h, dtype=float),
            where=views_24h > 0
        )

        # Engagement velocity per hour
        engagement_24h = likes_24h + comments_24h + shares_24h
        engagement_velocity_24h = engagement_24h / 24.0

        # Retention stats (in %)
        # Good videos ~ higher avg retention and quartiles
        base_retention = rng.normal(loc=45, scale=15, size=n_v)  # mean around 45%
        base_retention = base_retention.clip(10, 80)
        # correlate retention weakly with views success
        norm_views = np.log1p(views_24h)
        norm_views = (norm_views - norm_views.mean()) / (norm_views.std() + 1e-6)
        retention_avg_pct = (base_retention + 3 * norm_views).clip(5, 95)

        # Quartiles
        retention_p25 = (retention_avg_pct - rng.uniform(5, 15, size=n_v)).clip(0, 100)
        retention_p50 = (retention_avg_pct + rng.uniform(-5, 5, size=n_v)).clip(0, 100)
        retention_p75 = (retention_avg_pct + rng.uniform(5, 15, size=n_v)).clip(0, 100)

        # Virality label: views_24h > threshold * typical channel performance
        # Estimate each channel's median views_24h on the fly
        # We'll temporarily store and compute after loop; for now keep raw
        for i in range(n_v):
            emb = topic_emb[i]
            row = {
                "video_id": f"vid_{video_id_counter:07d}",
                "channel_id": channel_id,
                "niche": niche,
                "country": country,
                "subscriber_count": subs,
                "published_at": published_at[i],
                "day_of_week": int(day_of_week[i]),
                "upload_hour": int(upload_hour[i]),
                "title": titles[i],
                "title_length": int(title_lengths[i]),
                "thumbnail_text_length": int(thumb_text_len[i]),
                "video_duration_sec": int(durations[i]),
                "topic": topics[i],
                "sentiment_score": float(sentiment[i]),
                "views_24h": float(views_24h[i]),
                "views_7d": float(views_7d[i]),
                "likes_24h": int(likes_24h[i]),
                "comments_24h": int(comments_24h[i]),
                "shares_24h": int(shares_24h[i]),
                "like_ratio_24h": float(like_ratio_24h[i]),
                "comment_ratio_24h": float(comment_ratio_24h[i]),
                "engagement_velocity_24h": float(engagement_velocity_24h[i]),
                "retention_avg_pct": float(retention_avg_pct[i]),
                "retention_p25_pct": float(retention_p25[i]),
                "retention_p50_pct": float(retention_p50[i]),
                "retention_p75_pct": float(retention_p75[i]),
            }
            # add embedding dimensions
            for j in range(emb_dim):
                row[f"topic_emb_{j}"] = float(emb[j])

            rows.append(row)
            video_id_counter += 1

    videos = pd.DataFrame(rows)
    return videos


def add_virality_label(videos: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    # Compute per-channel median views_24h
    medians = videos.groupby("channel_id")["views_24h"].median().rename("channel_median_views_24h")
    videos = videos.merge(medians, on="channel_id", how="left")
    videos["virality_label"] = (
        videos["views_24h"] > (multiplier * videos["channel_median_views_24h"].clip(lower=1.0))
    ).astype(int)
    return videos.drop(columns=["channel_median_views_24h"])


def main(output_path: str, n_channels: int = 2500, n_videos: int = 120_000):
    rng = np.random.default_rng(RNG_SEED)

    print(f"Generating {n_channels} channels...")
    channels = generate_channels(n_channels, rng)

    print(f"Generating {n_videos} videos...")
    videos = generate_videos(channels, n_videos, rng)

    print("Adding virality label...")
    videos = add_virality_label(videos, multiplier=3.0)

    # Sort by time to simulate real data
    videos = videos.sort_values("published_at").reset_index(drop=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path} ...")
    videos.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="data/youtube_data.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2500,
        help="Number of channels"
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=120_000,
        help="Number of videos"
    )
    args = parser.parse_args()

    main(output_path=args.output,
         n_channels=args.channels,
         n_videos=args.videos)
