"""Tiny random Qwen2.5-VL CPU execution; does not download model weights."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
from probeforge.io import write_jsonl,read_jsonl,write_json
from contrastforge.vision import prepare_vision
from contrastforge.vision_train import train_vision
from contrastforge.collect import collect


def main():
    p=argparse.ArgumentParser();p.add_argument("--processor",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    root=Path(a.out).resolve()
    if root.exists():raise FileExistsError("Use a fresh output")
    root.mkdir();torch.set_num_threads(2);torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(a.processor,trust_remote_code=False,min_pixels=56*56,max_pixels=56*56)
    base=root/"tiny-base";processor.save_pretrained(base)
    cfg=Qwen2_5_VLConfig(vocab_size=152064,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
          num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048,
          eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id,
          image_token_id=151655,video_token_id=151656,vision_start_token_id=151652,
          vision_end_token_id=151653,rope_scaling={"type":"mrope","mrope_section":[1,1,2]},
          vision_config={"depth":2,"hidden_size":32,"intermediate_size":64,"num_heads":4,
                         "out_hidden_size":32,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2,
                         "window_size":112,"fullatt_block_indexes":[1]})
    model=Qwen2_5_VLForConditionalGeneration(cfg);model.save_pretrained(base)
    images=root/"images";images.mkdir();manifest=[]
    for i,split in enumerate(("train","validation","test")):
        image=Image.new("RGB",(100,100),(i*40,20,20));ImageDraw.Draw(image).rectangle((10,20,30,40),fill=(255,0,0));image.save(images/f"{i}.png")
        manifest.append({"image":f"{i}.png","scene_id":str(i),"split":split,"annotations_complete":True,
                         "vocabulary":["red_box"],"objects":[{"category":"red_box","bbox":[.1,.2,.3,.4]}]})
    write_jsonl(root/"annotations.jsonl",manifest);prepare_vision(root/"annotations.jsonl",images,root/"raw",["en"])
    for split in ("train","validation","test"):
        write_jsonl(root/f"{split}.jsonl",list(read_jsonl(root/f"raw/{split}.jsonl"))[:2])
    train_cfg={"model":str(base),"train_data":str(root/"train.jsonl"),"validation_data":str(root/"validation.jsonl"),
               "output_dir":str(root/"adapter"),"max_length":1024,"max_pixels":56*56,"allow_cpu":True,
               "lora_rank":4,"lora_alpha":8,"gradient_accumulation":1,"max_steps":1,"save_steps":1,"eval_steps":1}
    result=train_vision(train_cfg)
    collection=collect([root/"test.jsonl"],root/"collected.jsonl",model=str(base),adapter=str(root/"adapter"),vision=True,allow_cpu=True,budgets=[4])
    report={"passed":True,"kind":"tiny random Qwen2.5-VL CPU SFT, frozen vision, processor/adapter reload and generation",
            "metrics":result,"collection":collection,"not_tested":["3B CUDA training","quantized VLM","real annotated-image gains"]}
    write_json(root/"smoke_report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
