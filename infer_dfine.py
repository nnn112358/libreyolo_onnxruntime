#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime>=1.16.0", "opencv-python>=4.8.0", "numpy>=1.21.0"]
# ///
"""D-FINE ONNX 推論 (standalone, NMSフリー)。

family=dfine。出力 pred_logits[1,Q,nc] + pred_boxes[1,Q,4](cxcywh,[0,1])。
前処理: RGB / 単純square リサイズ / 255 / CHW (ImageNet正規化なし)。
後処理(DETR集合予測, NMS不要): sigmoid -> (Q*nc)をtop-k(<=300) -> cls=idx%nc,
query=idx//nc -> cxcywh->xyxy を元画像へスケール -> conf閾値。

  uv run infer_dfine.py
"""
import argparse, json, os
import cv2, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))


def color_for(c):
    np.random.seed(int(c) * 7 + 11)
    return tuple(int(v) for v in np.random.randint(64, 256, size=3))


def draw(img, dets, names):
    for x1, y1, x2, y2, sc, c in dets:
        col = color_for(c)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{names[c]} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), col, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def detr_parse(logits, boxes_cxcywh, W, H, conf, max_det=300):
    """DETR集合予測の共通デコード。NMSなし。"""
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    Q, nc = prob.shape
    flat = prob.reshape(-1)
    k = min(max_det, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    scores = flat[idx].astype(np.float32)
    query, cls = idx // nc, idx % nc
    cx, cy, w, h = boxes_cxcywh[query].T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    xyxy[:, [0, 2]] *= W; xyxy[:, [1, 3]] *= H
    dets = []
    for (x1, y1, x2, y2), sc, c in zip(xyxy, scores, cls):
        if sc <= conf:
            continue
        dets.append((int(np.clip(x1, 0, W)), int(np.clip(y1, 0, H)),
                     int(np.clip(x2, 0, W)), int(np.clip(y2, 0, H)), float(sc), int(c)))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "dfine_n.onnx"))
    ap.add_argument("--image", default=os.path.join(HERE, "image", "sample_640x480.jpg"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    out_path = a.out or os.path.splitext(a.image)[0] + "_dfine.jpg"

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map or {}
    imgsz = int(meta.get("imgsz", 640))
    nm = json.loads(meta["names"]); names = [nm[str(i)] for i in range(len(nm))]
    inp = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]

    img = cv2.imread(a.image)
    H, W = img.shape[:2]
    # 前処理: 単純square リサイズ + RGB + /255
    rgb = cv2.cvtColor(cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    blob = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

    outs = dict(zip(onames, sess.run(None, {inp: blob})))
    logits = outs["pred_logits"][0]
    boxes = outs["pred_boxes"][0]
    dets = detr_parse(logits, boxes, W, H, a.conf)

    draw(img, dets, names); cv2.imwrite(out_path, img)
    print(f"[dfine] model={os.path.basename(a.model)} imgsz={imgsz} image={W}x{H} detected={len(dets)} (NMSフリー)")
    for x1, y1, x2, y2, sc, c in sorted(dets, key=lambda d: -d[4]):
        print(f"  - {names[c]:<14} {sc:.3f} ({x1},{y1},{x2},{y2})")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
