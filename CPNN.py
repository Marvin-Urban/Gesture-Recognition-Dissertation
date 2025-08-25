import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

def _embeddings(X, y, subjects, reps, use_reps, n_classes):
    m = {}
    mask = np.isin(reps, use_reps)
    X, y, s = X[mask], y[mask], subjects[mask].astype(str)
    for sid in np.unique(s):
        rows = (s == sid)
        pcs = []
        for c in range(n_classes):
            rc = rows & (y == c)
            v = X[rc].mean(axis=0) if np.any(rc) else np.zeros(X.shape[1], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            pcs.append(v)
        m[sid] = np.concatenate(pcs, 0)
    return m

def _topk(target_sid, emb, k=5):
    t = emb[str(target_sid)]
    sims = []
    for sid, v in emb.items():
        if sid == str(target_sid): continue
        s = float(np.dot(t, v) / ((np.linalg.norm(t)+1e-12)*(np.linalg.norm(v)+1e-12)))
        sims.append((s, sid))
    sims.sort(reverse=True)
    return [sid for _, sid in sims[:k]]

def _source_graph(in_dim, n_classes=18, drop=0.3):
    x = layers.Input((in_dim,))
    y1 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    y1 = layers.Dropout(drop)(y1)
    y2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(y1)
    y2 = layers.Dropout(drop)(y2)
    y3 = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')(y2)
    y3 = layers.Dropout(drop)(y3)
    y_out = layers.Dense(n_classes, activation='softmax', dtype='float32')(y3)
    src_train = models.Model(x, y_out)
    src_feats = models.Model(x, [y1, y2, y3])
    return src_train, src_feats

def _target_from_frozen(src_feats, in_dim, n_classes=18, drop=0.3):
    for l in src_feats.layers: l.trainable = False
    x = layers.Input((in_dim,))
    s1, s2, s3 = src_feats(x)
    x1 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    a1 = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(s1)
    x1 = layers.Dropout(drop)(x1); x1 = layers.concatenate([x1, a1])
    x2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x1)
    a2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(s2)
    x2 = layers.Dropout(drop)(x2); x2 = layers.concatenate([x2, a2])
    x3 = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')(x2)
    a3 = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')(s3)
    x3 = layers.Dropout(drop)(x3); x3 = layers.concatenate([x3, a3])
    out = layers.Dense(n_classes, activation='softmax', dtype='float32')(x3)
    return models.Model(x, out)

def train_cpnn_5nn_frozen(
    X, y, subjects, reps, target_sid, *,
    epochs, batch_size,
    n_classes=18, k=5, val_size=0.2, seed=42
):
    in_dim = X.shape[1]
    emb = _embeddings(X, y, subjects, reps, use_reps=(1,2,3,4), n_classes=n_classes)
    nbr = _topk(target_sid, emb, k=k)

    src_mask = np.isin(subjects.astype(str), nbr)
    Xs, ys = X[src_mask], y[src_mask]
    Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(Xs, ys, test_size=val_size, stratify=ys, random_state=seed)

    src_train, src_feats = _source_graph(in_dim, n_classes=n_classes, drop=0.3)
    src_train.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    es = callbacks.EarlyStopping(min_delta=1e-3, patience=5, restore_best_weights=True)
    src_train.fit(Xs_tr, ys_tr, batch_size=batch_size, epochs=epochs,
                  validation_data=(Xs_val, ys_val), callbacks=[es], verbose=0)

    t_tr = (subjects == target_sid) & np.isin(reps, (1,2,3,4))
    t_te = (subjects == target_sid) & (reps == 5)
    Xt, yt = X[t_tr], y[t_tr]
    Xt_tr, Xt_val, yt_tr, yt_val = train_test_split(Xt, yt, test_size=val_size, stratify=yt, random_state=seed)

    tar = _target_from_frozen(src_feats, in_dim, n_classes=n_classes, drop=0.3)
    tar.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tar.fit(Xt_tr, yt_tr, batch_size=batch_size, epochs=epochs,
            validation_data=(Xt_val, yt_val), callbacks=[es], verbose=0)

    Xte, yte = X[t_te], y[t_te]
    return tar.evaluate(Xte, yte, batch_size=batch_size, verbose=0), tar, src_train, nbr
