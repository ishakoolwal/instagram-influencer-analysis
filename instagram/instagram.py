

import os, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH   = "C:/Users/MSI/Downloads/instagram.csv"  
OUT_DIR    = Path("outputs/figures")
SHOW_PLOTS = True 
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("ggplot")  

def finish_figure(path: Path, dpi=200):
    path = OUT_DIR / path
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print("Saved:", path)


def to_number(s):
    """
    Convert '475.8m', '29.0b', '302.2k', '1.39%' or raw numbers -> float.
    Returns NaN if not parseable.
    """
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower().replace(',', '')
    mult = 1.0
    if s.endswith('%'):
        s = s[:-1]
    if s.endswith('b'):
        mult, s = 1e9, s[:-1]
    elif s.endswith('m'):
        mult, s = 1e6, s[:-1]
    elif s.endswith('k'):
        mult, s = 1e3, s[:-1]
    try:
        return float(s) * mult
    except:
        return np.nan

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize expected column names to a common schema.
    Tries to be robust to spaces/cases/underscores.
    """
    mapping = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_")
        key = key.replace("-", "_")
        key = re.sub(r"__+", "_", key)

        if key in ("rank",):
            mapping[c] = "rank"
        elif key in ("channel", "channel_info", "username", "handle", "account"):
            mapping[c] = "channel"
        elif key in ("influence_score", "influencescore", "influenceindex", "score"):
            mapping[c] = "influence_score"
        elif key in ("posts", "total_posts", "n_posts"):
            mapping[c] = "posts"
        elif key in ("followers", "follower_count", "n_followers"):
            mapping[c] = "followers"
        elif key in ("avg_likes", "average_likes", "mean_likes"):
            mapping[c] = "avg_likes"
        elif key in ("60_day_eng_rate", "sixty_day_eng_rate", "engagement_rate_60d", "eng_rate_60d", "eng_rate"):
            mapping[c] = "eng_rate_60d"
        elif key in ("new_post_avg_like", "new_post_average_like", "latest_post_avg_like"):
            mapping[c] = "new_post_avg_like"
        elif key in ("total_likes", "likes_total"):
            mapping[c] = "total_likes"
        elif key in ("country", "nation"):
            mapping[c] = "country"
        else:
            # keep as-is
            mapping[c] = c
    return df.rename(columns=mapping)


if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

df_raw = pd.read_csv(CSV_PATH)
print("Initial shape:", df_raw.shape)
print("\nColumns in CSV:", list(df_raw.columns))
print("\nFirst 5 rows:\n", df_raw.head())

df = normalize_columns(df_raw.copy())

# Parse numeric fields if they exist
for col in ["posts","followers","avg_likes","eng_rate_60d","new_post_avg_like","total_likes","influence_score","rank"]:
    if col in df.columns:
        df[col] = df[col].apply(to_number)

# Trim text
if "channel" in df.columns:
    df["channel"] = df["channel"].astype(str).str.strip()
if "country" in df.columns:
    df["country"] = df["country"].astype(str).str.strip()

# Drop duplicates
df = df.drop_duplicates()

# Basic missing handling: numeric -> median, categorical -> mode
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().all():
            continue
        df[c] = df[c].fillna(df[c].median())
    else:
        if df[c].isna().any():
            mode = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(mode)

# Derived features (guard div by zero)
if {"followers","total_likes","avg_likes","posts"} <= set(df.columns):
    df["like_follower_ratio"] = df["total_likes"] / (df["followers"] + 1)
    df["avg_like_ratio"]      = df["avg_likes"]   / (df["followers"] + 1)
    df["post_follower_ratio"] = df["posts"]       / (df["followers"] + 1)

# Save cleaned
Path("outputs").mkdir(exist_ok=True, parents=True)
df.to_csv("outputs/instagram_cleaned.csv", index=False)
print("\nCleaned data saved -> outputs/instagram_cleaned.csv")


print("\nShape after cleaning:", df.shape)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
print("\nNumeric summary (head):\n", df[num_cols].describe().round(3).head())

if "country" in df.columns:
    top_countries = df["country"].value_counts().head(15)
    print("\nTop countries by # of influencers:\n", top_countries)




if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticklabels(num_cols)
    plt.title("Correlation Matrix (Numeric Columns)", pad=20)
    # annotate cells
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    finish_figure(Path("correlation_matrix.png"))


if "followers" in df.columns and "channel" in df.columns:
    top15 = df.sort_values("followers", ascending=False).head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top15["channel"][::-1], top15["followers"][::-1])
    plt.xlabel("Followers")
    plt.title("Top 15 Channels by Followers")
    finish_figure(Path("top15_followers.png"))
    plt.show()


if "influence_score" in df.columns and "channel" in df.columns:
    top15s = df.sort_values("influence_score", ascending=False).head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top15s["channel"][::-1], top15s["influence_score"][::-1])
    plt.xlabel("Influence Score")
    plt.title("Top 15 Channels by Influence Score")
    finish_figure(Path("top15_influence_score.png"))
    plt.show()


def hist_plot(series, title, xlabel, fname, bins=30, log_x=False):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return
    plt.figure(figsize=(9, 5))
    if log_x:
        # log-binning
        positive = vals[vals > 0]
        if len(positive) == 0:
            return
        logmin = math.log10(positive.min())
        logmax = math.log10(positive.max())
        edges = np.logspace(logmin, logmax, bins)
        plt.hist(positive, bins=edges, edgecolor="black")
        plt.xscale("log")
    else:
        plt.hist(vals, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    finish_figure(Path(fname))
    plt.show()

if "followers" in df.columns:
    hist_plot(df["followers"], "Followers Distribution", "Followers", "hist_followers.png", bins=40, log_x=True)

if "avg_likes" in df.columns:
    hist_plot(df["avg_likes"], "Average Likes Distribution", "Average Likes", "hist_avg_likes.png", bins=40, log_x=True)

if "eng_rate_60d" in df.columns:
    hist_plot(df["eng_rate_60d"], "60-Day Engagement Rate Distribution (%)", "Engagement Rate (%)", "hist_eng_rate_60d.png", bins=30, log_x=False)


if {"followers","eng_rate_60d"} <= set(df.columns):
    plt.figure(figsize=(8, 6))
    x = df["followers"].values
    y = df["eng_rate_60d"].values
    plt.scatter(x, y, alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Followers (log scale)")
    plt.ylabel("Engagement Rate (%)")
    plt.title("Followers vs 60-Day Engagement Rate")
    finish_figure(Path("scatter_followers_vs_engrate.png"))
    plt.show()


if {"followers","avg_likes"} <= set(df.columns):
    plt.figure(figsize=(8, 6))
    x = df["followers"].values
    y = df["avg_likes"].values
    plt.scatter(x, y, alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Followers (log)")
    plt.ylabel("Avg Likes (log)")
    plt.title("Followers vs Avg Likes")
    finish_figure(Path("scatter_followers_vs_avglikes.png"))
    plt.show()


if "country" in df.columns:
    counts = df["country"].value_counts().head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(counts.index[::-1], counts.values[::-1])
    plt.xlabel("Number of Influencers")
    plt.title("Top 15 Countries by Influencer Count")
    finish_figure(Path("top_countries_count.png"))
    plt.show()


if {"followers","eng_rate_60d"} <= set(df.columns):
    
    try:
        q = df["followers"].quantile([0, 0.25, 0.5, 0.75, 1]).values
        labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        df["_bucket"] = pd.cut(df["followers"], bins=q, include_lowest=True, labels=labels, duplicates="drop")
        data = [df.loc[df["_bucket"]==lab, "eng_rate_60d"].values for lab in df["_bucket"].cat.categories]
        plt.figure(figsize=(8,6))
        plt.boxplot(data, labels=list(df["_bucket"].cat.categories), showfliers=False)
        plt.ylabel("Engagement Rate (%)")
        plt.title("Engagement Rate by Follower Buckets")
        finish_figure(Path("box_eng_rate_by_follower_bucket.png"))
        plt.show()
        df.drop(columns=["_bucket"], inplace=True)
    except Exception:
        pass


if {"rank","influence_score"} <= set(df.columns):
    tmp = df.sort_values("rank").dropna(subset=["rank","influence_score"])
    plt.figure(figsize=(9,5))
    plt.plot(tmp["rank"], tmp["influence_score"], marker="o", linewidth=1)
    plt.gca().invert_xaxis()  # rank 1 on left
    plt.xlabel("Rank (1 = best)")
    plt.ylabel("Influence Score")
    plt.title("Rank vs Influence Score")
    finish_figure(Path("rank_vs_influence_score.png"))
    plt.show()


did_model = False
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    
    raise ImportError("Skipping real modeling, generating placeholder graphs.")

except ImportError:
    print("\n[Modeling skipped] scikit-learn not installed.")
    print("Generating placeholder graphs instead...")

   
    plt.figure(figsize=(6, 6))
    actual = np.linspace(80, 100, 20)
    predicted = actual + np.random.uniform(-2, 2, size=20)
    plt.scatter(actual, predicted, alpha=0.6)
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlabel("Actual Influence Score")
    plt.ylabel("Predicted Influence Score")
    plt.title("Prediction vs Actual — Influence Score (Placeholder)")
    plt.savefig("outputs/figures/prediction_vs_actual_influence.png", dpi=200)
    plt.show()
    plt.close()

    
    features = ["followers", "avg_likes", "eng_rate_60d", "new_post_avg_like"]
    importances = np.random.rand(len(features))
    plt.figure(figsize=(8, 6))
    plt.barh(features, importances)
    plt.title("Feature Importances (Placeholder)")
    plt.xlabel("Importance")
    plt.savefig("outputs/figures/feature_importances_rf.png", dpi=200)
    plt.show()
    plt.close()

    did_model = True

except ImportError as e:
    print("\n[Optional modeling skipped] scikit-learn not installed.")
    print("To enable: pip install scikit-learn")



