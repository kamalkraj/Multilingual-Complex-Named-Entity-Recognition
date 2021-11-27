import argparse
import os
import json


def readfile(filename):
    """
    Reads a file and returns a list of dictionaries.
    """
    data = []
    with open(filename, "r") as f:
        tokens = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("# id"):
                if len(tokens) > 0:
                    data.append({"tokens": tokens, "labels": labels})
                    tokens = []
                    labels = []
                continue
            splits = line.split(" ")
            tokens.append(splits[0])
            labels.append(splits[-1])

        if len(tokens) > 0:
            data.append({"tokens": tokens, "labels": labels})
    return data


def main():
    parser = argparse.ArgumentParser(description="Format the original training data to huggingface ner json format")
    parser.add_argument("--input_folder", type=str, default="training_data", help="input folder")
    parser.add_argument("--language_folder", type=str, default="EN-English", help="language folder")
    parser.add_argument("--output_folder", type=str, default="training_data_json", help="output folder")
    args = parser.parse_args()

    input_folder = args.input_folder
    language_folder = args.language_folder

    input_folder = os.path.join(input_folder, language_folder)
    output_folder = os.path.join(args.output_folder, language_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith("train.conll"):
            data = readfile(os.path.join(input_folder, filename))
            assert len(data) == 15300
            output_filename = os.path.join(output_folder, "train.json")
            with open(output_filename, "w") as f:
                for line in data:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        elif filename.endswith("dev.conll"):
            data = readfile(os.path.join(input_folder, filename))
            assert len(data) == 800
            output_filename = os.path.join(output_folder, "dev.json")
            with open(output_filename, "w") as f:
                for line in data:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        elif filename.endswith("test.conll"):
            data = readfile(os.path.join(input_folder, filename))
            output_filename = os.path.join(output_folder, "test.json")
            with open(output_filename, "w") as f:
                for line in data:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
