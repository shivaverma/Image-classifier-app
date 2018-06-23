import ast
import argparse
from time import time, sleep
from os import listdir
from classifier import classifier
from print_functions_for_lab_checks import *


def main():

    start_time = time()

    in_arg = get_input_args()
    image_dir = in_arg.dir
    model = in_arg.arch
    dogfile = in_arg.dogfile

    answers_dic = get_pet_labels(image_dir)

    result_dic = classify_images(image_dir, answers_dic, model)
    print(result_dic)
    print('\n')

    adjust_results4_isadog(result_dic, dogfile)
    print(result_dic)
    print('\n')

    results_stats_dic = calculates_results_stats(result_dic)

    print_results(result_dic, results_stats_dic, model)

    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time

    print("Total Elapsed Runtime: ", tot_time)


def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/')
    parser.add_argument('--arch', type=str, default='vgg')
    parser.add_argument('--dogfile', type=str, default='dognames.txt')
    return parser.parse_args()


def get_pet_labels(image_dir):

    """
        Creates a dictionary of pet labels based upon the filenames of the image
        files. Reads in pet filenames and extracts the pet image labels from the
        filenames and returns these label as petlabel_dic. This is used to check
        the accuracy of the image classifier model.
        Parameters:
         image_dir - The (full) path to the folder of images that are to be
                     classified by pretrained CNN models (string)
        Returns:
         petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                         Labels (as value)
    """

    list_files = listdir(image_dir)
    petlabels_dic = dict()

    for i in range(0, len(list_files)):

        if list_files[i][0] != '.':

            string = list_files[i][::-1]
            _, pet_label = string.split('_', 1)
            pet_label = pet_label[::-1]
            pet_label = pet_label.replace('_', ' ')
            petlabels_dic[list_files[i]] = pet_label

    return petlabels_dic


def classify_images(image_dir, petlabel_dic, model):

    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifier labels and 0 = no match between labels
    """

    result_dic = {}

    for i in petlabel_dic:

        truth = petlabel_dic[i]                       # petlabel corresponding to image directory
        truth = truth.lower()                         # converting it to lower case
        pred = classifier(image_dir+i, model)         # prediction of image file
        pred = pred.lower()                           # converting it to lower case
        found = 0

        if pred.find(truth) >= 0:                     # checking truth string in pred string

            if len(truth) == len(pred) - pred.find(truth) or pred[len(truth) + pred.find(truth)] in [' ', ',']:
                found = 1

        result_dic[image_dir+i] = [truth, pred, found]

    return result_dic


def adjust_results4_isadog(result_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifier labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """

    file = open(dogfile, 'r')
    dogname_dict = file.read().splitlines()

    for i in result_dic:

        is_truth_image_dog = 0
        is_pred_image_dog = 0

        for j in dogname_dict:

            temp = result_dic[i][0]
            if j.find(temp) >= 0:
                if len(temp) == len(j) - j.find(temp) or j[len(temp) + j.find(temp)] in [' ', ',']:
                    is_truth_image_dog = 1
                    break

        for j in dogname_dict:

            temp = result_dic[i][1]

            if ',' in temp:
                temp, _ = result_dic[i][1].split(',', 1)

            if j.find(temp) >= 0:
                if len(temp) == len(j) - j.find(temp) or j[len(temp) + j.find(temp)] in [' ', ',']:
                    is_pred_image_dog = 1
                    break

        result_dic[i].append(is_truth_image_dog)
        result_dic[i].append(is_pred_image_dog)


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """

    results_stats = dict()

    results_stats['total_images'] = 0
    results_stats['total_dog_images'] = 0
    results_stats['correct_pred_images'] = 0
    results_stats['correct_pred_dog_images'] = 0

    for i in results_dic:

        results_stats['total_images'] += 1
        if results_dic[i][2] == 1:
            results_stats['correct_pred_images'] += 1
        if results_dic[i][3] == 1:
            results_stats['total_dog_images'] += 1
        if results_dic[i][3] == 1 and results_dic[i][4] == 1:
            results_stats['correct_pred_dog_images'] += 1

    return results_stats


def print_results(results_dic, results_stats, model):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """

    print("Total images: " + str(results_stats['total_images']))
    print("Correctly classified images: " +
          str((results_stats['correct_pred_images']/results_stats['total_images'])*100) + '%')
    print("Total dog images: " + str(results_stats['total_dog_images']))
    print("Correctly clarified dog images: " +
          str((results_stats['correct_pred_dog_images'] / results_stats['total_dog_images']) * 100) + '%')
                

if __name__ == "__main__":

    main()
