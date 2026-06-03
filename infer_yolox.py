#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime>=1.16.0", "opencv-python>=4.8.0", "numpy>=1.21.0"]
# ///
"""YOLOX ONNX 推論 (standalone)。

family=yolox。出力 [1, N, 5+nc] = cxcywh(canvas px) + objectness + class(sigmoid)。
前処理: BGR 0-255 / 左上詰めletterbox(pad=114) / 正規化なし / CHW。
後処理: score = obj * max_cls -> conf -> NMS -> /ratio。

  uv run infer_yolox.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "yolox_n.onnx"))
    ap.add_argument("--image", default=os.path.join(HERE, "sample_640x480.jpg"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args()
    out_path = a.out or os.path.splitext(a.image)[0] + "_yolox.jpg"

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map or {}
    imgsz = int(meta.get("imgsz", 416))
    nm = json.loads(meta["names"]); names = [nm[str(i)] for i in range(len(nm))]
    inp = sess.get_inputs()[0].name

    img = cv2.imread(a.image)
    H, W = img.shape[:2]
    # 前処理: BGRのまま 左上詰め letterbox、正規化なし(0-255)
    r = min(imgsz / H, imgsz / W)
    nh, nw = int(H * r), int(W * r)
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(canvas.astype(np.float32).transpose(2, 0, 1)[None])

    pred = sess.run(None, {inp: blob})[0][0]            # (N, 5+nc)
    cxcywh, obj, clss = pred[:, :4], pred[:, 4], pred[:, 5:]
    conf = obj * clss.max(1); cls = clss.argmax(1)
    m = conf > a.conf
    cxcywh, conf, cls = cxcywh[m], conf[m], cls[m]

    dets = []
    if len(cxcywh):
        cx, cy, w, h = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)  # xyxy(canvas px)
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1], w, h])
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), a.conf, a.iou)
        for i in np.array(keep).flatten():
            x1, y1, x2, y2 = boxes[i] / r               # 元画像座標へ
            dets.append((int(np.clip(x1, 0, W)), int(np.clip(y1, 0, H)),
                         int(np.clip(x2, 0, W)), int(np.clip(y2, 0, H)), float(conf[i]), int(cls[i])))

    draw(img, dets, names); cv2.imwrite(out_path, img)
    print(f"[yolox] model={os.path.basename(a.model)} imgsz={imgsz} image={W}x{H} detected={len(dets)}")
    for x1, y1, x2, y2, sc, c in sorted(dets, key=lambda d: -d[4]):
        print(f"  - {names[c]:<14} {sc:.3f} ({x1},{y1},{x2},{y2})")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
