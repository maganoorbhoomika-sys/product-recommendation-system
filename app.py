import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================
# 2. CONFIGURATION
# ============================================================
RUN_SUFFIX = "_sample100000"
BEST_CF_WEIGHT = 0.5
TOP_K_RECS = 5
TOP_N_CLUSTER_PRODUCTS = 200

TRAIN_PATH = f"interactions_train{RUN_SUFFIX}.parquet"
FEATURES_PATH = f"user_features_final{RUN_SUFFIX}.parquet"
ASSIGN_PATH = f"user_cluster_assignments{RUN_SUFFIX}.parquet"
MAPPINGS_PATH = f"id_mappings{RUN_SUFFIX}.pkl"
SIM_IDX_PATH = f"item_top_n_indices{RUN_SUFFIX}.npy"
SIM_SCORE_PATH = f"item_top_n_scores{RUN_SUFFIX}.npy"
SPARSE_PATH = f"user_item_train_sparse{RUN_SUFFIX}.npz"

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
def check_required_files():
    required_files = {
        "Train data": TRAIN_PATH,
        "User features": FEATURES_PATH,
        "Cluster assignments": ASSIGN_PATH,
        "ID mappings": MAPPINGS_PATH,
        "Item similarity indices": SIM_IDX_PATH,
        "Item similarity scores": SIM_SCORE_PATH,
        "Sparse user-item matrix": SPARSE_PATH,
    }
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    return missing_files

def normalize_scores(score_dict):
    if not score_dict:
        return {}
    max_score = max(score_dict.values())
    if max_score <= 0:
        return score_dict
    return {item: score / max_score for item, score in score_dict.items()}

def get_user_profile(user_id, user_features):
    if "userId" not in user_features.columns:
        return None
    profile = user_features[user_features["userId"].astype(str) == str(user_id)]
    if profile.empty:
        return None
    return profile.iloc[0]

# ============================================================
# 4. LOAD ARTIFACTS
# ============================================================
@st.cache_resource(show_spinner="Loading recommendation system...")
def load_artifacts():
    missing_files = check_required_files()
    if missing_files:
        raise FileNotFoundError("The following deployment files are missing:\n" + "\n".join(missing_files))

    train_df = pd.read_parquet(TRAIN_PATH, columns=["userId", "productId", "Rating"])
    user_features = pd.read_parquet(FEATURES_PATH)
    cluster_assignments = pd.read_parquet(ASSIGN_PATH)

    with open(MAPPINGS_PATH, "rb") as file:
        mappings = pickle.load(file)
    user_to_idx = mappings["user_to_idx"]
    product_to_idx = mappings["product_to_idx"]
    idx_to_product = mappings["idx_to_product"]

    top_n_indices = np.load(SIM_IDX_PATH, mmap_mode="r")
    top_n_scores = np.load(SIM_SCORE_PATH, mmap_mode="r")
    user_item_train = sparse.load_npz(SPARSE_PATH).tocsr()

    product_popularity = train_df["productId"].value_counts().index.tolist()
    user_to_cluster = dict(zip(cluster_assignments["userId"], cluster_assignments["cluster"]))

    train_with_cluster = train_df.merge(cluster_assignments[["userId", "cluster"]], on="userId", how="inner")
    cluster_popularity = {}
    for cluster_id, group in train_with_cluster.groupby("cluster", observed=True):
        products = group["productId"].value_counts().index.tolist()
        cluster_popularity[cluster_id] = products[:TOP_N_CLUSTER_PRODUCTS]

    del train_with_cluster
    return {
        "train_df": train_df,
        "user_features": user_features,
        "cluster_assignments": cluster_assignments,
        "user_to_idx": user_to_idx,
        "product_to_idx": product_to_idx,
        "idx_to_product": idx_to_product,
        "top_n_indices": top_n_indices,
        "top_n_scores": top_n_scores,
        "user_item_train": user_item_train,
        "product_popularity": product_popularity,
        "cluster_popularity": cluster_popularity,
        "user_to_cluster": user_to_cluster,
    }

try:
    artifacts = load_artifacts()
except FileNotFoundError as error:
    st.error("Required deployment files are missing.")
    st.code(str(error))
    st.stop()
except Exception as error:
    st.error("The recommendation system could not be loaded.")
    st.exception(error)
    st.stop()

# ============================================================
# 5. RECOMMENDERS
# ============================================================
def recommend_popularity(rated_indices, product_popularity, product_to_idx, top_k=5):
    recommendations = []
    for product_id in product_popularity:
        if product_id not in product_to_idx:
            continue
        product_index = product_to_idx[product_id]
        if product_index in rated_indices:
            continue
        recommendations.append(product_id)
        if len(recommendations) >= top_k:
            break
    return recommendations

def recommend_item_cf(user_idx, user_item_train, top_n_indices, top_n_scores, idx_to_product, top_k=5):
    user_row = user_item_train.getrow(user_idx)
    rated_indices = set(user_row.indices)
    if len(rated_indices) == 0:
        return []
    user_mean = float(user_row.data.mean())
    item_scores = {}
    for item_index, rating in zip(user_row.indices, user_row.data):
        preference = float(rating - user_mean)
        if preference == 0:
            continue
        if item_index >= len(top_n_indices):
            continue
        similar_items = top_n_indices[item_index]
        similarity_scores = top_n_scores[item_index]
        for candidate_index, similarity in zip(similar_items, similarity_scores):
            candidate_index = int(candidate_index)
            similarity = float(similarity)
            if candidate_index in rated_indices:
                continue
            if similarity <= 0:
                continue
            score = similarity * preference
            item_scores[candidate_index] = item_scores.get(candidate_index, 0.0) + score

    if not item_scores:
        return []
    ranked_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = []
    for item_index, score in ranked_items:
        if item_index not in idx_to_product:
            continue
        product_id = idx_to_product[item_index]
        recommendations.append(product_id)
        if len(recommendations) >= top_k:
            break
    return recommendations

def recommend_cluster(cluster_id, rated_indices, cluster_popularity, product_to_idx, product_popularity, top_k=5):
    cluster_products = cluster_popularity.get(cluster_id, [])
    recommendations = []
    for product_id in cluster_products:
        if product_id not in product_to_idx:
            continue
        product_index = product_to_idx[product_id]
        if product_index in rated_indices:
            continue
        recommendations.append(product_id)
        if len(recommendations) >= top_k:
            break
    if len(recommendations) < top_k:
        fallback = recommend_popularity(rated_indices, product_popularity, product_to_idx, top_k * 2)
        for product_id in fallback:
            if product_id not in recommendations:
                recommendations.append(product_id)
            if len(recommendations) >= top_k:
                break
    return recommendations

def recommend_hybrid(user_id, user_idx, artifacts, cf_weight=0.5, top_k=5):
    user_item_train = artifacts["user_item_train"]
    top_n_indices = artifacts["top_n_indices"]
    top_n_scores = artifacts["top_n_scores"]
    idx_to_product = artifacts["idx_to_product"]
    product_to_idx = artifacts["product_to_idx"]
    cluster_popularity = artifacts["cluster_popularity"]
    user_to_cluster = artifacts["user_to_cluster"]
    product_popularity = artifacts["product_popularity"]

    user_row = user_item_train.getrow(user_idx)
    rated_indices = set(user_row.indices)

    cf_raw_scores = {}
    user_mean = float(user_row.data.mean()) if len(user_row.data) > 0 else 0.0
    for item_index, rating in zip(user_row.indices, user_row.data):
        preference = float(rating - user_mean)
        if preference == 0:
            continue
        if item_index >= len(top_n_indices):
            continue
        similar_items = top_n_indices[item_index]
        similarities = top_n_scores[item_index]
        for candidate_index, similarity in zip(similar_items, similarities):
            candidate_index = int(candidate_index)
            similarity = float(similarity)
            if candidate_index in rated_indices:
                continue
            if similarity <= 0:
                continue
            score = similarity * preference
            cf_raw_scores[candidate_index] = cf_raw_scores.get(candidate_index, 0.0) + score

    cf_raw_scores = {item: score for item, score in cf_raw_scores.items() if score > 0}
    cluster_raw_scores = {}
    cluster_id = user_to_cluster.get(user_id)
    if cluster_id is not None:
        cluster_products = cluster_popularity.get(cluster_id, [])
        total_products = len(cluster_products)
        if total_products > 0:
            for rank, product_id in enumerate(cluster_products):
                if product_id not in product_to_idx:
                    continue
                product_index = product_to_idx[product_id]
                if product_index in rated_indices:
                    continue
                score = (total_products - rank) / total_products
                cluster_raw_scores[product_index] = score

    cf_scores = normalize_scores(cf_raw_scores)
    cluster_scores = normalize_scores(cluster_raw_scores)

    candidate_items = set(cf_scores.keys()) | set(cluster_scores.keys())
    if not candidate_items:
        return recommend_popularity(rated_indices, artifacts["product_popularity"], product_to_idx, top_k)

    final_scores = {}
    for item_index in candidate_items:
        cf_score = cf_scores.get(item_index, 0.0)
        cluster_score = cluster_scores.get(item_index, 0.0)
        final_scores[item_index] = cf_weight * cf_score + (1 - cf_weight) * cluster_score

    ranked_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = []
    for item_index, score in ranked_items:
        if item_index not in idx_to_product:
            continue
        product_id = idx_to_product[item_index]
        recommendations.append(product_id)
        if len(recommendations) >= top_k:
            break

    if len(recommendations) < top_k:
        fallback = recommend_popularity(rated_indices, product_popularity, product_to_idx, top_k * 2)
        for product_id in fallback:
            if product_id not in recommendations:
                recommendations.append(product_id)
            if len(recommendations) >= top_k:
                break
    return recommendations

# ============================================================
# 6. SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🛍️ Recommendation System")
    st.markdown("**Project:** Cluster-augmented Recommendations")
    st.divider()
    st.write(f"**CF Weight:** {BEST_CF_WEIGHT:.2f}")
    st.write(f"**Recommendations (K):** {TOP_K_RECS}")
    st.write(f"**Users:** {len(artifacts['user_to_idx']):,}")
    st.write(f"**Products:** {len(artifacts['product_to_idx']):,}")

# ============================================================
# 7. MAIN PAGE
# ============================================================
st.title("🛍️ Recommendation System")

# --- MULTI-CLUSTER BROWSER SECTION ---
st.subheader("👥 Explore All Behavioral Clusters")
st.markdown("Browse product recommendations tailored for each specific user segment/cluster directly:")

all_clusters = sorted(list(artifacts["cluster_popularity"].keys()))
selected_cluster = st.selectbox("Select Cluster to Explore", all_clusters, format_func=lambda x: f"Cluster {x}")

if selected_cluster is not None:
    cluster_recs = recommend_cluster(
        selected_cluster,
        set(),
        artifacts["cluster_popularity"],
        artifacts["product_to_idx"],
        artifacts["product_popularity"],
        TOP_K_RECS
    )
    cols = st.columns(len(cluster_recs))
    for i, product_id in enumerate(cluster_recs):
        with cols[i]:
            st.metric(label=f"Recommendation #{i+1}", value=product_id)

st.divider()

# --- USER PROFILE & HYBRID RECS SECTION ---
st.subheader("Enter User ID for Personalized Recommendations")
selected_user_id = st.text_input("User ID", placeholder="Enter a valid User ID").strip()

if selected_user_id:
    user_idx = artifacts["user_to_idx"].get(selected_user_id)
    if user_idx is None:
        for key, value in artifacts["user_to_idx"].items():
            if str(key) == selected_user_id:
                user_idx = value
                break

    if user_idx is None:
        st.warning("This User ID is not present in the training data. Showing popular products instead.")
        cold_start = recommend_popularity(set(), artifacts["product_popularity"], artifacts["product_to_idx"], TOP_K_RECS)
        for num, pid in enumerate(cold_start, start=1):
            st.write(f"**{num}.** {pid}")
    else:
        st.subheader(f"Recommendations for User: {selected_user_id}")
        user_profile = get_user_profile(selected_user_id, artifacts["user_features"])
        cluster_id = artifacts["user_to_cluster"].get(selected_user_id, "N/A")

        col1, col2, col3 = st.columns(3)
        col1.metric("User ID", selected_user_id)
        if user_profile is not None:
            rating_count = user_profile.get("n_ratings", 0)
            col2.metric("Number of Ratings", f"{int(rating_count):,}")
        else:
            col2.metric("Number of Ratings", "N/A")
        col3.metric("Assigned Cluster", str(cluster_id))

        hybrid_recommendations = recommend_hybrid(selected_user_id, user_idx, artifacts, cf_weight=BEST_CF_WEIGHT, top_k=TOP_K_RECS)

        st.subheader(f"⭐ Top {TOP_K_RECS} Hybrid Recommendations")
        for num, pid in enumerate(hybrid_recommendations, start=1):
            st.write(f"**{num}.** {pid}")

        st.divider()
        st.subheader("Recommendation Methods Comparison")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🔗 Item-Item CF")
            cf_recs = recommend_item_cf(user_idx, artifacts["user_item_train"], artifacts["top_n_indices"], artifacts["top_n_scores"], artifacts["idx_to_product"], TOP_K_RECS)
            for num, pid in enumerate(cf_recs, start=1):
                st.write(f"{num}. {pid}")

        with col_right:
            st.markdown("### 👥 Cluster-Based")
            user_row = artifacts["user_item_train"].getrow(user_idx)
            rated_indices = set(user_row.indices)
            clust_recs = recommend_cluster(cluster_id, rated_indices, artifacts["cluster_popularity"], artifacts["product_to_idx"], artifacts["product_popularity"], TOP_K_RECS)
            for num, pid in enumerate(clust_recs, start=1):
                st.write(f"{num}. {pid}")

st.divider()
st.caption("Product Recommendation System | Item-Item Collaborative Filtering + Multi-Cluster Selection")
