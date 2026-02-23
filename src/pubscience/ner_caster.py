# Source: https://github.com/bramiozo/PubScience/blob/main/pubscience/share/ner_caster.py

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class TAGS(BaseModel):
    start: int
    end: int
    tag: str


class NER(BaseModel):
    """
    {'tags': [{'start': 19, 'end': 32, 'tag': 'PER'},
              {'start': 6, 'end': 32, 'tag': 'LOC'}],
     'id': 'blaId',
     'text': "bladiebla"}
    """

    tags: List[TAGS]
    id: str
    text: str


class NameMap(BaseModel):
    id: str
    tag: str
    start: str
    end: str


class NERFormer:
    """
    Transform incoming formats in the NER format
    Incoming formats:
        (*.ann, *.txt) -> NER
        (db.tsv, *.txt) -> NER

    Outgoing formats:
        NER -> (*.ann, *.txt)
        NER -> (db.tsv, *.txt)
    """

    def __init__(
        self,
        ann_dir: str,
        txt_dir: str,
        db_path: str | None,
        out_path: str | None,
        name_map: NameMap | None,
        write_to_file: bool = True,
    ):

        self.ann_dir = ann_dir
        self.txt_dir = txt_dir
        self.db_path = db_path
        self.out_path = out_path

        # check if (ann_dir exists and txt_dir) or (db_path exists and txt_dir)
        # if not raise ValueError
        if (ann_dir is None and db_path is None) or txt_dir is None:
            raise ValueError("Please provide a valid ann/db/txt directory")

        if isinstance(db_path, str):
            if not os.path.exists(db_path):
                raise ValueError("Please provide a valid db_path")
            if name_map is None:
                name_map = NameMap(
                    **{
                        "id": "filename",
                        "tag": "label",
                        "start": "start_span",
                        "end": "end_span",
                    }
                )
                print("Continuing with default name map")

        if out_path is not None:
            print(f"You set the out_path. We are writing to {self.out_path}")
            write_to_file = True
            self.out_path = out_path
        else:
            if write_to_file:
                raise ValueError("Please provide an output directory")

        self.write_to_file = write_to_file
        self.name_map = name_map

    def _text_adder(self, tag_dict: Dict[str, List[TAGS]]) -> List[NER]:
        output_jsonl = []
        for k, v in tag_dict.items():
            # get the text from the text file
            file_name = os.path.join(self.txt_dir, f"{k}.txt")
            with open(file_name, "r", encoding="utf-8") as fread:
                text = fread.read()
            output_jsonl.append(NER(tags=v, id=k, text=text))
        return output_jsonl

    def parse_db(self, db_path: str) -> List[NER]:
        # read file
        # first line contains header with tab-separated names
        with open(db_path, "r", encoding="utf-8") as fread:
            lines = fread.readlines()

        id_str = self.name_map.id
        tag_str = self.name_map.tag
        start_str = self.name_map.start
        end_str = self.name_map.end

        # get the header
        # get the index of the text column
        header = lines[0].strip().split("\t")
        print(header)

        res_dict = defaultdict(list)
        for r in lines[1:]:
            if r.strip() == "":
                continue
            rdict = dict(zip(header, r.strip().split("\t")))
            TAG = TAGS(
                start=int(rdict[start_str]), end=int(rdict[end_str]), tag=rdict[tag_str]
            )
            res_dict[rdict[id_str]].append(TAG)

        # second iteration to add the text, we only need to to this once per id
        output_jsonl = self._text_adder(res_dict)
        return output_jsonl

    def parse_ann(self, ann_list: List[str]) -> List[NER]:
        res_dict = defaultdict(list)
        for ann in ann_list:
            # read the ann file
            # get the text from the text file
            file_name = os.path.join(self.ann_dir, ann)
            with open(file_name, "r", encoding="utf-8") as fread:
                lines = fread.readlines()

            for l in lines:
                # parse the line
                # example; T1	DISEASE 188 200	ritmestormen
                # now, id= ann tag = DISEASE, start = 188, end = 200
                if l.startswith("#"):
                    continue
                try:
                    l = l.strip().split("\t")
                    tag = l[1].split(" ")[0]
                    start = int(l[1].split(" ")[1])
                    end = int(l[1].split(" ")[2])
                    res_dict[ann.replace(".ann", "")].append(
                        TAGS(start=start, end=end, tag=tag)
                    )
                except Exception as e:
                    print(f"Error {e}, for {ann}, problems with {l}")
                    if l[1].startswith("AnnotatorNotes"):
                        continue
                    print(f"Error in parsing {l}")

        output_jsonl = self._text_adder(res_dict)
        return output_jsonl

    def transform(self) -> List[NER] | None:
        # read the list of files in the directory for ann/txt
        #
        txt_list = os.listdir(self.txt_dir)

        if self.ann_dir is not None:
            ann_list = [f for f in os.listdir(self.ann_dir) if f.endswith(".ann")]
            out_jsonl = self.parse_ann(ann_list)
        elif self.db_path is not None:
            out_jsonl = self.parse_db(self.db_path)

        if self.write_to_file:
            # write the output to the out_path
            with open(self.out_path, "w") as f:
                for line in out_jsonl:
                    f.write(json.dumps(line.model_dump()) + "\n")

        if not self.write_to_file:
            return out_jsonl


def collect_jsons(
    jsons_path,
    out_path,
    entity_map={
        "ENFERMEDAD": "DISEASE",
        "FARMACO": "MEDICATION",
        "SINTOMA": "SYMPTOM",
        "PROCEDIMIENTO": "PROCEDURE",
    },
):
    # go through jsonl's and per "id" collect/extend the "tags" lists
    #
    id_tags_map = {}
    for js in os.listdir(jsons_path):
        # load jsonl into list
        if js.endswith(".json"):
            with open(os.path.join(jsons_path, js), "r") as f:
                jsonl_list = [json.loads(line) for line in f]

            for json_obj in jsonl_list:
                id = json_obj["id"]
                tags = json_obj["tags"]
                text = json_obj["text"]
                if id in id_tags_map:
                    id_tags_map[id]["tags"].extend(tags)
                else:
                    id_tags_map[id] = {"tags": tags, "text": text}

    if isinstance(entity_map, dict):
        # 'tag' have to be mapped
        for id, data in id_tags_map.items():
            for tag in data["tags"]:
                tag["tag"] = entity_map.get(tag["tag"], tag["tag"])

    # remove duplicate {'tag': xx, 'start': xx, 'end': }
    for id, data in id_tags_map.items():
        span_tuples_set = set()
        new_list = []
        for t in data["tags"]:
            _t = (t["tag"], t["start"], t["end"])
            if _t in span_tuples_set:
                continue
            span_tuples_set.add(_t)
            new_list.append(t)
        data["tags"] = new_list

    # write to jsonl
    with open(out_path, "w") as f:
        for id, data in id_tags_map.items():
            json.dump({"id": id, "tags": data["tags"], "text": data["text"]}, f)
            f.write("\n")

    # provide counts on the tags lists [{'tag': xxx..}..]
    tag_counts = defaultdict(int)
    for id, data in id_tags_map.items():
        for tag in data["tags"]:
            tag_counts[tag["tag"]] += 1

    print("Tag counts:")
    for tag, count in tag_counts.items():
        print(f"{tag}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform NER data")
    parser.add_argument("--ann_dir", type=str, help="Directory with ann files")
    parser.add_argument("--txt_dir", type=str, help="Directory with txt files")
    parser.add_argument(
        "--json_dir", type=str, help="Directory that contains partial annotation json's"
    )
    parser.add_argument("--db_path", type=str, help="Path to db file")
    parser.add_argument("--out_path", type=str, help="Path to output file")
    parser.add_argument("--name_map", type=str, help="Mapping of column names")
    parser.add_argument("--collect_json", action="store_true")
    parser.add_argument("--write_to_file", action="store_true")

    args = parser.parse_args()
    try:
        name_map = json.loads(args.name_map)
    except:
        name_map = None

    if not args.collect_json:
        ner = NERFormer(
            ann_dir=args.ann_dir,
            txt_dir=args.txt_dir,
            db_path=args.db_path,
            out_path=args.out_path,
            name_map=name_map,
            write_to_file=args.write_to_file,
        )
        ner.transform()
    else:
        collect_jsons(args.json_dir, args.out_path)
