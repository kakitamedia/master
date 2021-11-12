import argparse
import json

from copy import deepcopy

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_FILE = './datasets/coco/annotations/instances_minival2014.json'
size_ranges = [10*i for i in range(70)]

def parse_args():
    parser = argparse.ArgumentParser(description='pytorch training code')
    parser.add_argument('--results', type=str, default='', help='')

    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.results) as f:
        results = json.load(f)

    with open(GT_FILE) as f:
        gt = json.load(f)
    
    for i in range(len(size_ranges)-1):
        size_range = [size_ranges[i], size_ranges[i+1]]
        print(size_range)
        filtered_results = size_filter(results, size_range)
        filtered_gt = size_filter(gt, size_range)

        with open('./results.json', 'w') as f:
            json.dump(filtered_results, f)
        
        with open('./gt.json', 'w') as f:
            json.dump(filtered_gt, f)

        if len(filtered_results) == 0 or len(filtered_gt['annotations']) == 0:
            print('no boxes are founded')
            continue

        coco_gt = COCO('./gt.json')
        coco_dt = coco_gt.loadRes('./results.json')
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
    # for result in results:
    #     print(result['bbox'])

def size_filter(file, size_range):
    filtered = []
    if type(file) == dict:
        temp = file['annotations']
    else:
        temp = file

    for ann in temp:
        w, h = ann['bbox'][2:]
        size = max(w, h)
        if size_range[0] <= size < size_range[1]:
            filtered.append(ann)

    if type(file) == dict:
        temp = deepcopy(file)
        temp['annotations'] = filtered
        return temp
    else:
        return filtered

if __name__ == '__main__':
    main()
