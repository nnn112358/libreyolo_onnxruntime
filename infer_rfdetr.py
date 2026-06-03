#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime>=1.16.0", "opencv-python>=4.8.0", "numpy>=1.21.0"]
# ///
"""RF-DETR ONNX 推論 (standalone, NMSフリー)。

family=rfdetr。他のDETRと異なる点:
  - 入力名は "input"、入力サイズはサイズ毎に異なる(n=384)。
  - 前処理: RGB / 単純square リサイズ / 255 / ImageNet正規化。
  - 出力: dets[1,300,4](cxcywh,[0,1]) + labels[1,300,91]  ← 91クラス(COCO91)。
  - 後処理: sigmoid -> top-k(300) -> COCO91->80 へ写像(対象外は破棄) -> 元画像へスケール。
    NMSなし。

  uv run infer_rfdetr.py
"""
import argparse, json, os
import cv2, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# COCO91(1始まり) -> COCO80(0始まり)。RF-DETRのlabelインデックスはこの91クラス系。
COCO91_TO_80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
    14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
    24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
    37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
    58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
    72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
    82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "rfdetr_n.onnx"))
    ap.add_argument("--image", default=os.path.join(HERE, "image", "sample_640x480.jpg"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--num-select", type=int, default=300)
    a = ap.parse_args()
    out_path = a.out or os.path.splitext(a.image)[0] + "_rfdetr.jpg"

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map or {}
    imgsz = int(meta.get("imgsz", 384))
    nm = json.loads(meta["names"]); names = [nm[str(i)] for i in range(len(nm))]
    inp = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]

    img = cv2.imread(a.image)
    H, W = img.shape[:2]
    # 前処理: 単純square リサイズ + RGB + /255 + ImageNet正規化
    rgb = cv2.cvtColor(cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    blob = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    outs = dict(zip(onames, sess.run(None, {inp: blob})))
    boxes = outs["dets"][0]      # (300, 4) cxcywh [0,1]
    logits = outs["labels"][0]   # (300, 91)

    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    Q, nc = prob.shape
    flat = prob.reshape(-1)
    k = min(a.num_select, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    scores = flat[idx].astype(np.float32)
    query, cls91 = idx // nc, idx % nc

    dets = []
    for q, c91, sc in zip(query, cls91, scores):
        if sc <= a.conf:
            continue
        c80 = COCO91_TO_80.get(int(c91), -1)   # COCO91 -> COCO80 (対象外は破棄)
        if c80 < 0:
            continue
        cx, cy, w, h = boxes[q]
        x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        dets.append((int(np.clip(x1, 0, W)), int(np.clip(y1, 0, H)),
                     int(np.clip(x2, 0, W)), int(np.clip(y2, 0, H)), float(sc), c80))

    draw(img, dets, names); cv2.imwrite(out_path, img)
    print(f"[rfdetr] model={os.path.basename(a.model)} imgsz={imgsz} input='{inp}' image={W}x{H} detected={len(dets)} (NMSフリー/91->80)")
    for x1, y1, x2, y2, sc, c in sorted(dets, key=lambda d: -d[4]):
        print(f"  - {names[c]:<14} {sc:.3f} ({x1},{y1},{x2},{y2})")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
