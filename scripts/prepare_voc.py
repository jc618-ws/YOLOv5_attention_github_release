from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import sys

ROOT = Path("/root/CBAM-YOLOv5")
VOC = ROOT / "datasets/VOC2007/VOCdevkit/VOC2007"
OUT = ROOT / "datasets/VOC2007/YOLO"

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

SPLITS = ["train", "val", "test"]

def read_ids(split):
    main = VOC / "ImageSets/Main"
    reference = None
    result = None

    for cls in CLASSES:
        file = main / f"{cls}_{split}.txt"
        if not file.exists():
            raise FileNotFoundError(file)

        ids = [
            line.split()[0]
            for line in file.read_text().splitlines()
            if line.strip()
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(f"{file.name} 存在重复编号")

        current = set(ids)

        if reference is None:
            reference = current
            result = ids
        elif current != reference:
            raise ValueError(f"{file.name} 图片编号不一致")

    return result

def convert_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    if width <= 0 or height <= 0:
        raise ValueError(f"图片尺寸错误: {xml_path}")

    labels = []
    difficult_count = 0

    for obj in root.findall("object"):
        name = obj.findtext("name")

        if name not in CLASSES:
            raise ValueError(f"未知类别 {name}: {xml_path}")

        difficult = int(obj.findtext("difficult", "0"))

        # 按常见 VOC-YOLO 转换约定排除 difficult=1
        if difficult == 1:
            difficult_count += 1
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError(f"缺少 bndbox: {xml_path}")

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        # 限制边界框在图片范围内
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"无效边界框: {xml_path}")

        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        cls_id = CLASSES.index(name)

        labels.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return labels, difficult_count

def main():
    image_dir = VOC / "JPEGImages"
    annotation_dir = VOC / "Annotations"

    images = {}
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
            if p.stem in images:
                raise ValueError(f"重复图片编号: {p.stem}")
            images[p.stem] = p

    xmls = {p.stem: p for p in annotation_dir.glob("*.xml")}

    split_ids = {s: read_ids(s) for s in SPLITS}

    trainval = set(read_ids("trainval"))
    train = set(split_ids["train"])
    val = set(split_ids["val"])
    test = set(split_ids["test"])

    if train | val != trainval or train & val:
        raise ValueError("train、val、trainval 划分关系错误")

    if trainval & test:
        raise ValueError("trainval 与 test 存在数据泄漏")

    expected = trainval | test

    if set(images) != expected:
        raise ValueError("图片与数据集划分不一致")

    if set(xmls) != expected:
        raise ValueError("XML 与数据集划分不一致")

    print("数据集完整性检查通过")

    OUT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        image_out = OUT / "images" / split
        label_out = OUT / "labels" / split

        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        total_objects = 0
        difficult_total = 0
        empty_labels = 0

        for image_id in split_ids[split]:
            image_path = images[image_id]
            xml_path = xmls[image_id]

            # 使用软链接，不复制原始图片
            link_path = image_out / image_path.name

            if link_path.is_symlink():
                if link_path.resolve() != image_path.resolve():
                    raise ValueError(f"软链接指向错误: {link_path}")
            elif link_path.exists():
                raise FileExistsError(link_path)
            else:
                link_path.symlink_to(image_path.resolve())

            labels, difficult = convert_xml(xml_path)

            label_path = label_out / f"{image_id}.txt"
            label_path.write_text(
                "\n".join(labels) + ("\n" if labels else "")
            )

            total_objects += len(labels)
            difficult_total += difficult

            if not labels:
                empty_labels += 1

        print(
            f"{split}: 图片={len(split_ids[split])}, "
            f"目标={total_objects}, "
            f"空标签={empty_labels}, "
            f"排除difficult={difficult_total}"
        )

    # 生成 YOLOv5 数据集配置
    config = {
        "path": str(OUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": CLASSES
    }

    yaml_path = OUT / "VOC.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config, f, allow_unicode=True, sort_keys=False
        )

    print("\n数据集转换完成！")
    print("配置文件:", yaml_path)
    print("类别数量:", len(CLASSES))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("转换失败:", e)
        sys.exit(1)
