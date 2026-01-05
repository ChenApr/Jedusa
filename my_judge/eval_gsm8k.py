import json
import re
import argparse
import sys

def extract_answer_number(text):
    # Try to find #### first
    if "####" in text:
        text = text.split("####")[-1]
    
    # Remove commas from numbers (e.g. 1,234 -> 1234)
    text = text.replace(",", "")
    
    # Find all numbers (integers or floats)
    # This regex matches integers and floats
    numbers = re.findall(r"-?\d+\.?\d*", text)
    
    if not numbers:
        return None
    
    # Return the last number found
    return float(numbers[-1])

def eval_gsm8k(answer_file, ground_truth_file):
    print(f"Evaluating {answer_file} against {ground_truth_file}")
    
    # Load ground truth
    gts = {}
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            gts[item["question_id"]] = item
            
    # Load model answers
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            answers[item["question_id"]] = item
            
    correct = 0
    total = 0
    
    for qid, gt_item in gts.items():
        if qid not in answers:
            print(f"Warning: Question {qid} not found in answers.")
            continue
            
        total += 1
        
        gt_text = gt_item["answer"]
        gt_num = extract_answer_number(gt_text)
        
        model_output = answers[qid]["choices"][0]["turns"][0]
        model_num = extract_answer_number(model_output)
        
        if gt_num is not None and model_num is not None and abs(gt_num - model_num) < 1e-6:
            correct += 1
        
    if total == 0:
        print("No matching questions found.")
        return
        
    accuracy = correct / total
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save result
    result_file = answer_file.replace(".jsonl", "_result.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Total: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    print(f"Result saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--ground-truth-file", type=str, required=True)
    args = parser.parse_args()
    
    eval_gsm8k(args.answer_file, args.ground_truth_file)
