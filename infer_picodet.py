#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime>=1.16.0", "opencv-python>=4.8.0", "numpy>=1.21.0"]
# ///
"""PicoDet ONNX 推論 (standalone)。

family=picodet。出力 [1, N, 4+nc] = xyxy(canvas px) + class(sigmoid)。
前処理: RGB / 単純リサイズ(letterboxなし) / ImageNet正規化(0-255空間) / CHW。
後処理: conf -> NMS -> 単純リサイズの逆スケール(orig/imgsz)。

  uv run infer_picodet.py
"""
import argparse, json, os
import cv2, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = np.array([123.675, 116.28, 103.53], np.float32)   # 0-255スケール
STD = np.array([58.395, 57.12, 57.375], np.float32)


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
    ap.add_argument("--model", default=os.path.join(HERE, "picodet_s.onnx"))
    ap.add_argument("--image", default=os.path.join(HERE, "image", "sample_640x480.jpg"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args()
    out_path = a.out or os.path.splitext(a.image)[0] + "_picodet.jpg"

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map or {}
    imgsz = int(meta.get("imgsz", 320))
    nm = json.loads(meta["names"]); names = [nm[str(i)] for i in range(len(nm))]
    inp = sess.get_inputs()[0].name

    img = cv2.imread(a.image)
    H, W = img.shape[:2]
    # 前処理: 単純リサイズ + RGB + ImageNet正規化(0-255空間)
    rgb = cv2.cvtColor(cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) - MEAN) / STD
    blob = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    pred = sess.run(None, {inp: blob})[0][0]            # (N, 4+nc)
    boxes, scores = pred[:, :4], pred[:, 4:]            # xyxy(canvas px), cls(sigmoid)
    conf = scores.max(1); cls = scores.argmax(1)
    m = conf > a.conf
    boxes, conf, cls = boxes[m], conf[m], cls[m]

    dets = []
    if len(boxes):
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), a.conf, a.iou)
        sx, sy = W / imgsz, H / imgsz                   # 単純リサイズの逆スケール
        for i in np.array(keep).flatten():
            x1, y1, x2, y2 = boxes[i]
            dets.append((int(np.clip(x1 * sx, 0, W)), int(np.clip(y1 * sy, 0, H)),
                         int(np.clip(x2 * sx, 0, W)), int(np.clip(y2 * sy, 0, H)), float(conf[i]), int(cls[i])))

    draw(img, dets, names); cv2.imwrite(out_path, img)
    print(f"[picodet] model={os.path.basename(a.model)} imgsz={imgsz} image={W}x{H} detected={len(dets)}")
    for x1, y1, x2, y2, sc, c in sorted(dets, key=lambda d: -d[4]):
        print(f"  - {names[c]:<14} {sc:.3f} ({x1},{y1},{x2},{y2})")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
