from app import app, db
from models import Petition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# clustering.py
def cluster_petitions():
    petitions = Petition.query.filter(Petition.content_text.isnot(None)).all()
    if not petitions:
        return {}
    try:
        texts = [p.content_text for p in petitions]
        petition_ids = [p.id for p in petitions]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        n_clusters = min(max(2, len(petitions) // 20), 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(tfidf_matrix)
        
        clusters = {}
        for idx, label in enumerate(kmeans.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'petition_id': petition_ids[idx],
                'title': petitions[idx].title,  # Include title for display
                'distance': np.linalg.norm(tfidf_matrix[idx].toarray() - kmeans.cluster_centers_[label])
            })
        
        # Convert cluster identifiers from bytes to integers if applicable
        clusters_integer = {}
        for cluster_id, petition_list in clusters.items():
            if isinstance(cluster_id, bytes):
                cluster_id = int.from_bytes(cluster_id, byteorder='big')
            clusters_integer[cluster_id] = petition_list
        
        return clusters_integer  # Return the modified clusters

    except Exception as e:
        app.logger.error(f"Clustering error: {str(e)}")
        db.session.rollback()
        return {}
