"""Image-specific annotations and exact count/side-count supervision."""
import base64
import io
from fractions import Fraction as F
from pathlib import Path
from probeforge.io import canonical,digest,file_hash,read_jsonl,write_jsonl,write_json
from .languages import LANGUAGES,index


def image_for(row):
    from PIL import Image,ImageOps
    path=Path(row["image_path"])
    if file_hash(path)!=row["image_sha256"]:raise ValueError("Image changed since dataset preparation")
    with Image.open(path) as im:image=im.convert("RGB")
    return ImageOps.mirror(image) if row["image_transform"]=="horizontal_flip" else image


def expected(row):
    names=row["vocabulary"]
    if row["task"]=="object_counts":
        out={k:0 for k in names}
        for obj in row["objects"]:out[obj["category"]]+=1
    else:
        out={side:{k:0 for k in names} for side in ("left","center","right")}
        for obj in row["objects"]:
            x=(F(str(obj["bbox"][0]))+F(str(obj["bbox"][2])))/2
            if row["image_transform"]=="horizontal_flip":x=1-x
            side="left" if x<F(45,100) else "right" if x>F(55,100) else "center"
            out[side][obj["category"]]+=1
    return out


def gold_vision(row):return {"counts":expected(row)}


def verify_vision(row,obj):
    correct=set(obj)=={"counts"} and obj["counts"]==expected(row)
    # Python considers True==1; reject booleans and non-integer count leaves.
    def leaves(x):
        if isinstance(x,dict):return all(leaves(v) for v in x.values())
        return type(x) is int and x>=0
    correct=correct and leaves(obj.get("counts"))
    return {"correct":correct,"reason":"verified_image_annotations" if correct else "count_or_side_mismatch"}


def prepare_vision(manifest_path,image_root,out,languages=None):
    root=Path(out)
    if root.exists() and any(root.iterdir()):raise FileExistsError("Output exists")
    image_root=Path(image_root).resolve();languages=languages or list(LANGUAGES)
    splits={s:[] for s in ("train","validation","test")};scenes={};images={};seen=set()
    for source in read_jsonl(manifest_path):
        split=source["split"];scene=str(source["scene_id"])
        if split not in splits:raise ValueError("Use train, validation or test")
        if scenes.setdefault(scene,split)!=split:raise ValueError("Scene crosses splits")
        if source.get("annotations_complete") is not True:
            raise ValueError("Explicit complete annotations are required; missing labels cannot be interpreted as absent objects")
        p=(image_root/source["image"]).resolve()
        if not p.is_relative_to(image_root):raise ValueError("Image escapes image_root")
        sha=file_hash(p)
        if images.setdefault(sha,split)!=split:raise ValueError("Identical image crosses splits")
        if sha in seen:raise ValueError("Duplicate image annotation record")
        seen.add(sha)
        vocab=source["vocabulary"]
        if not vocab or len(vocab)!=len(set(vocab)) or any(not isinstance(x,str) for x in vocab):raise ValueError("Invalid vocabulary")
        for obj in source["objects"]:
            if obj["category"] not in vocab:raise ValueError("Object outside vocabulary")
            box=obj["bbox"]
            if len(box)!=4 or not all(type(x) in (int,float) and 0<=x<=1 for x in box) or not (box[0]<box[2] and box[1]<box[3]):
                raise ValueError("Bounding boxes must be normalized [xmin,ymin,xmax,ymax]")
        group=digest(["scene",scene])[:24]
        for transform in ("identity","horizontal_flip"):
            for task in ("object_counts","side_counts"):
                for lang in languages:
                    request=["Count the annotated object categories in the image.","Cuenta las categorías de objetos indicadas en la imagen.",
                             "Compta les categories d'objectes indicades a la imatge.","احسب أعداد فئات الأجسام المحددة في الصورة.",
                             "Ziel déi uginn Objetkategorien am Bild."][index(lang)]
                    spec=("Return JSON {\"counts\": {category: integer}}." if task=="object_counts" else
                          "Return JSON {\"counts\": {\"left\": {category: integer}, \"center\": {category: integer}, \"right\": {category: integer}}}. "
                          "Use bounding-box center x: left if x<0.45, right if x>0.55, center otherwise; x is normalized image width.")
                    row={"id":digest([sha,scene,transform,task,lang])[:24],"group_id":group,
                         "semantic_id":digest([sha,transform,task]),"family":"vision_"+task,"split":split,
                         "kind":"vlm","language":lang,"view":"base" if transform=="identity" else "reflection",
                         "image_path":str(p),"image_sha256":sha,"image_transform":transform,"scene_id":scene,
                         "task":task,"vocabulary":vocab,"objects":source["objects"],
                         "prompt":[{"role":"user","content":request+"\n"+spec+" Include every category, using zero for absent objects. Categories: "+canonical(vocab)}]}
                    splits[split].append(row)
    for split,rows in splits.items():
        if not rows:raise ValueError("Provide at least one annotated scene in each split")
        write_jsonl(root/f"{split}.jsonl",rows)
    report={"images":len(images),"scenes":len(scenes),"split_rows":{k:len(v) for k,v in splits.items()},
            "scope":"exact annotated counts and horizontal spatial equivariance; not driving safety or general scene understanding",
            "portable_paths":"Regenerate preparation after moving image_root to another machine"}
    write_json(root/"manifest.json",report);return report


def api_messages(row):
    image=image_for(row);buf=io.BytesIO();image.save(buf,format="PNG")
    uri="data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    return [{"role":"user","content":[{"type":"image_url","image_url":{"url":uri}},
                                       {"type":"text","text":row["prompt"][0]["content"]}]}]
