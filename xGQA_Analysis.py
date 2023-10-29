
#take /data/tir/projects/tir6/general/piyushkh/xGQA/CCLM/configs/cclm-base-ft/GQA.yaml as the argument and load in argparser

import argparse
import os
import sys
import yaml
import json
import matplotlib.pyplot as plt
from collections import Counter


config_file_path = "configs/cclm-base-ft/GQA.yaml"


# CCLM/CCLM_output/result/vqa_bn_epoch3_rank2.json

with open(config_file_path) as f:
    config = yaml.safe_load(f)

lang_list = config["test_file"].keys()

ans2label = json.load(open(config["answer_list"]))

error_analysis = {}

all_question_types = set()

for lang in lang_list:
    print("Evaluating for language: ", lang)
    incorrect_question_type = []  #test["types"]["structural"]
    incorrect_image_ids = []
    incorrect_questions = []
    incorrect_full_answers = []
    prediction_path = f"CCLM_output/result/vqa_{lang}_epoch3_rank0.json"
    ground_truth_path = config["test_file"][lang][0]
    
    print("prediction_path: ", prediction_path)
    print("ground_truth_path: ", ground_truth_path)
    #load prediction and ground truth jsons
    prediction_json = json.load(open(prediction_path))
    ground_truth_json = json.load(open(ground_truth_path))

    question_type_stats={}

    for question_id in ground_truth_json.keys():
        answer_prediction = prediction_json[question_id]["answer"]
        question_type = ground_truth_json[question_id]["types"]["structural"]
        if question_type not in question_type_stats:
            question_type_stats[question_type] = {"total": 0, "incorrect": 0}
        question_type_stats[question_type]["total"] += 1
        if answer_prediction != ground_truth_json[question_id]["answer"]:
            question_type_stats[question_type]["incorrect"] += 1
            incorrect_question_type.append(question_type)
            incorrect_image_ids.append(ground_truth_json[question_id]["image_id"])
            incorrect_questions.append(ground_truth_json[question_id]["question"])
            incorrect_full_answers.append(ground_truth_json[question_id]["answer"])
            all_question_types.add(ground_truth_json[question_id]["types"]["structural"])

    for question_type in question_type_stats.keys():
        question_type_stats[question_type]["accuracy"] = 1 - question_type_stats[question_type]["incorrect"]/question_type_stats[question_type]["total"]

    error_analysis[lang] = {"incorrect_question_type": incorrect_question_type, "incorrect_image_ids": incorrect_image_ids, "incorrect_questions": incorrect_questions, "incorrect_full_answers": incorrect_full_answers, "question_type_stats": question_type_stats}


#Save error analysis
with open("CCLM/CCLM_output/result/error_analysis.json", "w") as f:
    json.dump(error_analysis, f)

def plot_accuracy_for_all_question_types_for_all_lang(error_analysis):
    langs = error_analysis.keys()
    plt.plot(langs, [error_analysis[lang]["question_type_stats"][question_type]["accuracy"] for lang in langs])
    plt.title("Accuracy for all question types for all languages")
    plt.xlabel("Language")
    plt.ylabel("Accuracy")
    plt.savefig("CCLM/CCLM_output/result/accuracy_for_all_question_types_for_all_lang.png")
    plt.clf()

plot_accuracy_for_all_question_types_for_all_lang(error_analysis)


            






# def get_incorrect_question_types_acc(error_analysis):
#     question_type_accuracy = {}
#     for question_type in all_question_types:
#         total_count = 0
#         for gt in ground_truth_json.values():
#             if gt["types"]["structural"] == question_type:
#                 if gt["answer"] == prediction_json[gt["question_id"]]["answer"]:
#                     question_type_accuracy[question_type] += 1
#                 total_count += 1
#         question_type_accuracy[question_type] = question_type_accuracy[question_type]/total_count
#     return question_type_accuracy
    

def plot_incorrect_question_type_acc(error_analysis):
    lang_to_question_type_acc = {}
    for lang in lang_list:
        incorrect_question_type_acc = get_incorrect_question_types_acc(error_analysis[lang]["incorrect_question_type"])
        lang_to_question_type_acc[lang] = incorrect_question_type_acc
    #Plot line chart in matplotlibe with langs on x axis and question type accuracy on y axis
    for question_type in all_question_types:
        plt.plot(lang_list, [lang_to_question_type_acc[lang][question_type] for lang in lang_list])
        plt.title(f"Incorrect Question Type Accuracy for {question_type}")
        plt.xlabel("Language")
        plt.ylabel("Accuracy")
        plt.savefig(f"CCLM/CCLM_output/result/{question_type}_incorrect_question_type_acc.png")
        plt.clf()

        



                




        

    
    
    


